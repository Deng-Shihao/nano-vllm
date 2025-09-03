# 这份代码实现了一个“支持 KV Cache 的注意力(Attention)”模块，并配合 FlashAttention 与 Triton 内核，
# 在“预填充(prefill)”与“解码(decode)”两个阶段高效地读写/使用 K/V 缓存。
# 代码主要分为三块：
#   1) Triton 核函数 store_kvcache_kernel：把本步算出的 K/V 写入到分页化的 KV 缓存中
#   2) Python 侧包装函数 store_kvcache：做形状/步长检查并发射 Triton 内核
#   3) Attention(nn.Module)：前向逻辑，区分 prefill 与 decode，调用 FlashAttention 的可变长/带缓存两种路径

import torch
from torch import nn
import triton
import triton.language as tl

# flash_attn_varlen_func：可变长度(多序列拼接)的 FlashAttention 前向
# flash_attn_with_kvcache：解码阶段，直接在 KV Cache 上做注意力
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# get_context() 提供运行时上下文(例如：是否是 prefill、每条序列的长度前缀和、block table、slot 映射等)
from nanovllm.utils.context import get_context


# ---- 1) Triton 内核：把 K/V 写入 KV 缓存 --------------------------------------
@triton.jit
def store_kvcache_kernel(
    key_ptr,  # [N, num_kv_heads, head_dim] 扁平后的一行起始指针(按 stride 访问)
    key_stride,  # key 在第0维(样本维/N维)上的跨度：从第 i 行到第 i+1 行的偏移
    value_ptr,  # 同理，value 的数据指针
    value_stride,  # value 在第0维上的跨度
    k_cache_ptr,  # KV 缓存(K)的基指针，布局为 [num_slots, D]，D = num_kv_heads * head_dim
    v_cache_ptr,  # KV 缓存(V)的基指针
    slot_mapping_ptr,  # 长度为 N 的数组：把第 i 个 token 映射到 “缓存中的 slot 编号”
    D: tl.constexpr,  # 常量：单个 token 的“展平后 KV 宽度”，即 num_kv_heads * head_dim
):
    # 每个程序实例(program)负责处理一个 token（即一行 K/V）
    idx = tl.program_id(0)  # 获取当前并行实例在 0 号维度上的索引 -> [0, N)

    # 查到这个 token 对应的缓存 slot (分页/块式KV缓存常见做法：把逻辑位置映射到物理 slot)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    # 计算该 token 在 K/V 源张量中的元素偏移范围：
    #   行起点 = idx * stride
    #   行内列偏移 = 0..D-1
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 从源 K/V 中向量化读出这一整行(包含所有 kv-head 的 head_dim)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 计算该 slot 在缓存中的写入区间 [slot*D, slot*D + D)
    cache_offsets = slot * D + tl.arange(0, D)

    # 把当前步的 K/V 写入缓存
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# ---- 2) Python 侧包装：检查与发射内核 -----------------------------------------
def store_kvcache(
    key: torch.Tensor,  # 形状 [N, num_kv_heads, head_dim]
    value: torch.Tensor,  # 形状 [N, num_kv_heads, head_dim]
    k_cache: torch.Tensor,  # 形状 [num_slots, D]，D = num_kv_heads * head_dim
    v_cache: torch.Tensor,  # 形状同上
    slot_mapping: torch.Tensor,  # 形状 [N]，把每个 token -> slot
):
    # 从 key 的形状获取 N、num_kv_heads 与 head_dim
    N, num_heads, head_dim = key.shape
    D = (
        num_heads * head_dim
    )  # 对应 Triton 内核里的 D (注意：这里的 num_heads 实际上是 num_kv_heads)

    # 下述断言保证内存布局便于 Triton 连续矢量化读写(最后一维要连续)
    assert (
        key.stride(-1) == 1 and value.stride(-1) == 1
    ), "key/value 的最后一维必须是连续内存 (stride(-1) == 1)"

    # 确保 [N, H, D_head] 的布局在 H 维上的 stride 恰好是 head_dim，
    # 也就是同一 token 下不同 head 的数据是紧邻拼接的。
    assert (
        key.stride(1) == head_dim and value.stride(1) == head_dim
    ), "key/value 的第1维 stride 必须等于 head_dim"

    # 缓存的第1维(列)跨度应为 D(= num_kv_heads * head_dim)，即每个 slot 存下整行 KV
    assert (
        k_cache.stride(1) == D and v_cache.stride(1) == D
    ), "k_cache/v_cache 的列跨度必须等于 num_kv_heads * head_dim"

    # slot_mapping 的元素个数必须等于 N(一一对应每个 token)
    assert slot_mapping.numel() == N, "slot_mapping 的长度必须等于 N"

    # 以 (N,) 作为 Triton 网格大小：每个程序实例处理一个 token
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


# ---- 3) 注意力模块 ------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        num_heads,  # Q 的 head 数(注意：可能大于 KV 的 head 数，用于 GQA)
        head_dim,  # 每个 head 的维度
        scale,  # softmax 的缩放因子，通常为 1/sqrt(head_dim)
        num_kv_heads,  # K/V 的 head 数(GQA：num_kv_heads <= num_heads)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        # k_cache/v_cache 由外部在合适时机分配，这里用空 tensor 作为“未初始化”的标记
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        预期输入形状约定(常见约定之一)：
          - q: [N, num_heads * head_dim] 或已能 reshape 为 [-1, num_heads, head_dim]
          - k,v: 与 q 对齐，但 KV 使用 num_kv_heads
        该函数内部把它们 reshape 成 [N, H, Dh] 的三维张量，后续交给 FlashAttention。
        """
        o: torch.Tensor

        # 把最后一维还原成 [num_heads, head_dim] / [num_kv_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)  # [N, Hq, Dh]
        k = k.view(-1, self.num_kv_heads, self.head_dim)  # [N, Hkv, Dh]
        v = v.view(-1, self.num_kv_heads, self.head_dim)  # [N, Hkv, Dh]

        # 从全局/线程上下文拿到运行期信息(是否 prefill、变长拼接用的前缀和、block table、slot 映射等)
        context = get_context()

        # 当前模块持有的 KV 缓存句柄(可能为空)
        k_cache, v_cache = self.k_cache, self.v_cache

        # 如果缓存已存在(非空)，把本步算出的 K/V 写入到缓存中
        #   - prefill 时：把 prompt 的每个 token 的 KV 依次写入
        #   - decode 时：把新生成的下一个 token 的 KV 追加写入
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            # 预填充阶段：通常把一批序列拼接成一条长序列，用 cu_seqlens_* 来标识每条序列的边界
            if context.block_tables is not None:  # 存在 prefix cache(分页/块式缓存)
                # 若使用“前缀缓存”，那么直接让注意力读取缓存中的 K/V，
                # 这样可以避免重复读写/拷贝，提升吞吐。
                k, v = k_cache, v_cache

            # 变长 FlashAttention：
            #   q, k, v 可以是拼接后的平铺形式；
            #   cu_seqlens_q/k 给出每条序列的起止位置前缀和；
            #   max_seqlen_* 是本批次内的最大序列长度(用于内核调度)；
            #   causal=True 表示因果掩码；block_table 指示分页缓存的页映射。
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            # 解码阶段：每个序列一次只解一个 token，查询长度=1
            # q 当前是 [N, Hq, Dh]，为了满足常见接口 [B, T, H, Dh]，在 dim=1 处插入 T=1
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),  # [N, 1, Hq, Dh]
                k_cache,  # 在缓存里查历史的 K
                v_cache,  # 在缓存里查历史的 V
                cache_seqlens=context.context_lens,  # 每条序列已缓存的上下文长度(解码步数)
                block_table=context.block_tables,  # 分页缓存的页表
                softmax_scale=self.scale,
                causal=True,
            )

        # 输出回并行化后的二维/扁平形状：[N, Hq*Dh]
        o = o.view(-1, self.num_heads * self.head_dim)
        return o


# ----------------------------- 补充说明(要点总结) ------------------------------
# 1) GQA 支持：
#    - 模块允许 num_heads != num_kv_heads（如 8 个 Q-head，对应 4 个 KV-head）。
#    - 在写缓存时 D = num_kv_heads * head_dim，缓存按 KV 的头数展开保存。
#
# 2) 步长与布局检查：
#    - key/value 的最后一维必须是连续内存(stride(-1) == 1)，便于 Triton 向量化加载。
#    - key/value 在第1维上的 stride == head_dim，确保不同 head 在内存中紧邻排列。
#    - k_cache/v_cache 在列维度上的 stride == D，确保每个 slot 是完整一行 KV。
#
# 3) slot_mapping 与 block_tables：
#    - slot_mapping：把“逻辑上的第 i 个 token”映射到“物理缓存中的第 slot 行”，
#      常见于分页化 KV 缓存(例如 vLLM 的 PagedAttention)，用于碎片/复用管理。
#    - block_tables：描述每个序列使用到哪些物理页(块)，FlashAttention 内核据此去读缓存。
#
# 4) prefill vs decode：
#    - prefill：处理提示词(Prompt)阶段，序列一般较长，常用 varlen 接口并用 cu_seqlens_* 标识切分。
#    - decode：自回归生成阶段，每步只新增一个 token，直接用 flash_attn_with_kvcache 在缓存上做注意力。
#
# 5) scale：
#    - 通常取 1/sqrt(head_dim)，在 softmax 前对点积做缩放，稳定数值。
#
# 6) 资源/健壮性提示：
#    - 代码里没有分配 k_cache/v_cache，需外部按 [num_slots, num_kv_heads*head_dim] 预分配在 GPU 上，
#      且 dtype/device 要与计算一致。
#    - 若 decode 阶段缓存未初始化，flash_attn_with_kvcache 可能报错；实际系统中应确保先做过 prefill，
#      或显式初始化空缓存。
#    - 若需要跨步增长缓存，slot_mapping 与 context.context_lens 也要同步更新。
