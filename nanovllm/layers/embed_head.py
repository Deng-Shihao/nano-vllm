import torch  # 导入 PyTorch（张量、算子等）
from torch import nn  # 导入神经网络模块基类和组件
import torch.nn.functional as F  # 导入函数式 API（embedding, linear 等无状态操作）
import torch.distributed as dist  # 导入分布式通信模块（NCCL/Gloo 等后端）

from nanovllm.utils.context import (
    get_context,
)  # 从项目中导入上下文读取接口（用于 prefill/decode 时共享元信息）


class VocabParallelEmbedding(nn.Module):
    # 按词表维度做张量并行的 embedding 层：
    # 把词表按 tp_size 等分，每个进程/设备只保存自己那一份词表权重片段。
    # 前向时只查本分片的词向量，然后通过 all_reduce 汇总成完整的 embedding。

    def __init__(
        self,
        num_embeddings: int,  # 全词表大小（总的 token 数量）
        embedding_dim: int,  # 每个 token 的嵌入维度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()  # 当前张量并行进程的 rank（0..tp_size-1）
        self.tp_size = dist.get_world_size()  # 张量并行总进程数（分片数）

        assert num_embeddings % self.tp_size == 0  # 确保词表能被等分
        self.num_embeddings = num_embeddings  # 保存全局词表大小

        self.num_embeddings_per_partition = (
            self.num_embeddings // self.tp_size
        )  # 每个分片包含的词元数量（整除保证每分片大小相同）

        # n_embd(d_model): 768 (hidden dimension)
        # n_head: 12
        # head_dim = n_embd / n_head  = 64 表示每个注意力头（Attention Head）所分到的子空间维度

        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        # 本分片在全局词表中的起始 token id（inclusive）

        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 本分片在全局词表中的结束 token id（exclusive）

        # 本分片实际存储的权重形状 = [num_embeddings_per_partition, embedding_dim]
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        # 将一个自定义的权重加载器函数绑定到 weight 上，外部的模型加载器（load_model）会调用这个 loader
        # 用于把从公共权重文件读出来的全量权重切片并拷贝到 param 中
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 自定义的权重分片加载函数（供外部的权重加载器调用）
        # param: 本地的 param（形状是分片大小）
        # loaded_weight: 从文件/权重来源一次性加载的全量权重（形状是全词表大小）
        param_data = param.data  # 获取 param 的底层张量（用于就地 copy）
        shard_size = param_data.size(0)  # 本地分片在第 0 维的大小
        start_idx = self.tp_rank * shard_size  # 在全量权重中本分片的起始偏移
        # 使用 narrow 从全量权重中裁出本分片对应的连续区间（不做拷贝，只是视图）
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()  # 确认形状匹配
        param_data.copy_(loaded_weight)  # 把分片内容拷贝到 param（就地覆盖）

    def forward(self, x: torch.Tensor):
        # x: 输入的 token id 张量，常见形状有 [batch, seq_len] 或 [N]（LongTensor）
        # 返回：与 x 对应的 embedding 张量，形状为 x.shape + (embedding_dim,)

        # 如果开启了张量并行（tp_size > 1），需要把全局 token id 映射到本分片的本地索引
        if self.tp_size > 1:
            # mask 表示哪些输入 token 属于当前分片的词表范围（True/False）
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将全局 id 减去本分片的起始 id，得到本地 id（仅对 mask=True 的位置有效）
            x = mask * (x - self.vocab_start_idx)

        # 使用函数式 embedding 查表（会根据 x 的值在 weight 中索引）
        # 若 x 中有超出本分片范围的 id（被 mask 为 0），embedding 会返回对应索引 0 的向量，
        # 但后面我们会用 mask 把这些位置置为 0（见下）
        y = F.embedding(x, self.weight)  # y 形状 = x.shape + (embedding_dim,)

        if self.tp_size > 1:
            # 对于不属于当前分片的 token，embedding 得到的值是无意义的（因为 x 被改为 0 或负值）
            # 使用 mask 把这些位置置为 0（先扩展一维以匹配嵌入的维度）
            y = mask.unsqueeze(1) * y  # mask.unsqueeze(1) 形状扩展为 x.shape + (1,)

            # 将每张卡上局部的 embedding 用 all_reduce 求和，合并成完整的 embedding
            # 思路：每张卡只在自己词表分片对应的位置保留真实 embedding（其他位置为 0），
            # all_reduce(SUM) 后即可得到所有分片寄存的完整 embedding。
            dist.all_reduce(
                y
            )  # 默认 op=SUM，将结果写回 y（每个进程执行后 y 都是完整 embedding）

        return y  # 返回最终 embedding（每个元素为 embedding_dim 向量）


class ParallelLMHead(VocabParallelEmbedding):
    # LM head：把 hidden states 映射到词表 logits 的并行实现。
    # 该类继承 VocabParallelEmbedding，以复用分片权重及加载逻辑，并把 forward 反向实现为线性映射。

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)  # 调用父类初始化（分片权重等）
        if bias:
            # 如果需要 bias（按分片存储），为每个分片创建一个偏置向量
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            # 同样把自定义加载函数挂到 bias 上，供外部加载器使用
            self.bias.weight_loader = self.weight_loader
        else:
            # 无 bias 时需要显式注册 None（便于后续参数扫描/保存）
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        # x: 隐状态张量，通常形状为 [N, hidden_dim] 或 [batch*seq, hidden_dim]
        # 返回：在 rank0 上返回完整 vocab-size 的 logits（其它 rank 返回 None）
        # 注意：并行策略是每张卡只计算自己词表片段的 logits，然后在 rank0 上 gather 拼接成完整 logits。

        context = (
            get_context()
        )  # 读取线程/调用上下文（由 ModelRunner 的 set_context 设置）
        if context.is_prefill:
            # prefill 模式下，输入 x 常常是把整个 batch 的所有 token flatten（比如 prefill 把所有 prompt token 拼成一个长向量）
            # context.cu_seqlens_q 是 query 序列的前缀和（形式例如 [0, len1, len1+len2, ...]）
            # 取 cu_seqlens_q[1:] - 1，得到每个序列最后一个 query token 在扁平数组中的索引
            last_indices = context.cu_seqlens_q[1:] - 1
            # 选取每个序列最后一个位置的隐状态作为要投射到词表的输入（因为 prefill 通常在一次性计算后只需要这些位置进行后续 decode）
            x = x[
                last_indices
            ].contiguous()  # contiguous() 保证内存是连续的（便于后续线性运算）

        # 在每张卡上用分片权重做线性映射（等价于局部的 logits = x @ W_local^T + b_local）
        # F.linear 会自动处理 bias=None 的情况
        logits = F.linear(
            x, self.weight, self.bias
        )  # logits 形状 = [N, num_embeddings_per_partition]

        if self.tp_size > 1:
            # 如果并行，收集所有分片的 logits 到 rank0，并在 rank0 上拼接成完整词表的 logits：
            # - 首先，rank0 需要提供一个列表用于接收其它 rank 的张量。
            # - dist.gather 在所有 rank 上调用；在 dst=0 上将收到来自每个 rank 的 logits。
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            # 调用 dist.gather：把每个 rank 的 logits 发送到 rank0（rank0 的 all_logits 将被填充）
            dist.gather(logits, all_logits, 0)
            # 在 rank0 上，把按分片收集到的 logits 列表按词表维度拼接成完整 logits（dim=-1）
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
            # 注意：在非 0 rank，logits 被设为 None（因为没有完整词表的 logits），只有 rank0 返回完整 logits

        return logits  # 返回：rank0 上是完整 vocab_size logits；其他 rank 返回 None
