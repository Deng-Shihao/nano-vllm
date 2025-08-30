import pickle  # Python 内置序列化模块：把对象转为字节流/从字节流恢复（用于共享内存通信）
import torch  # PyTorch 主包
import torch.distributed as dist  # 分布式通信（NCCL/Gloo 等后端），用于多 GPU / 多进程并行
from multiprocessing.synchronize import Event  # 进程间同步原语 Event（由 mp.get_context("spawn") 创建）
from multiprocessing.shared_memory import SharedMemory  # 跨进程共享内存（零拷贝共享字节缓冲）

from nanovllm.config import Config  # 项目内：全局配置（模型路径、并行规模、内存参数等）
from nanovllm.engine.sequence import Sequence  # 序列对象（跟踪 prompt/生成状态/KV cache 布局）

from nanovllm.models.models import model_dict # 注册模型

# from nanovllm.models.qwen3 import Qwen3ForCausalLM  # 具体模型实现（Qwen3 因果语言模型）
# from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM # 具体模型实现（Qwen3moe 因果语言模型）

from nanovllm.layers.sampler import Sampler  # 采样器（温度/top-k/top-p 等策略，从 logits 采样下一 token）

# 上下文工具：把本批次的 FlashAttention/KV cache 相关元信息（slot_mapping、block_tables 等）注册到“线程局部上下文”
from nanovllm.utils.context import set_context, get_context, reset_context

# 权重加载工具（把权重从磁盘/缓存加载到模型）
from nanovllm.utils.loader import load_model


# 负责：初始化分布式/模型与显存、准备输入张量、执行前向、采样、以及（可选）CUDA Graph 复用
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config  # 保存配置对象
        hf_config = config.hf_config  # HF 配置（隐藏层数、头维度、精度 dtype 等）
        self.block_size = config.kvcache_block_size  # KV 缓存的“块”大小（每块能存多少个 token 的 KV）
        self.enforce_eager = config.enforce_eager  # 是否强制 Eager（禁用 CUDA Graph）
        self.world_size = config.tensor_parallel_size  # 张量并行（TP）规模 = 参与进程/GPU 数
        self.rank = rank  # 本进程的 rank（0 为主进程）
        self.event = event  # 主进程用 list[Event] 通知从进程；从进程用单个 Event 等待

        # 初始化分布式进程组：使用 NCCL 后端，通过 TCP 初始化地址/端口；指定总进程数与当前 rank
        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )

        torch.cuda.set_device(rank)  # 绑定当前进程到对应 GPU（rank 即 CUDA 设备号的约定）
        default_dtype = (torch.get_default_dtype())  # 记录当前默认 dtype，稍后临时切换到模型 dtype，再恢复
        torch.set_default_dtype(hf_config.torch_dtype)  # 将 PyTorch 默认 dtype 切到模型配置（例如 torch.float16/bfloat16）
        torch.set_default_device("cuda")  # 将默认 device 设为 CUDA：后续新建张量默认分配到 GPU

        self.model = model_dict[hf_config.model_type](hf_config)  # init Qwen3  # 构建模型结构（此时参数未加载）

        load_model(self.model, config.model)  # 加载权重到 GPU（可能是从 HF repo/本地路径）

        self.sampler = Sampler()  # 构建采样器（把 logits → token_id）

        self.warmup_model()  # 预热模型：跑一次虚拟输入，触发 cudnn/cublas 初始化，稳定首次延迟
        self.allocate_kv_cache()  # 按可用显存计算可分配的 KV cache 块数，并绑定到各层

        if (not self.enforce_eager):  # 如果不强制 Eager，则提前捕获 CUDA Graph（复用计算图降低 launch 开销）
            self.capture_cudagraph()

        torch.set_default_device("cpu")  # 把默认 device 切回 CPU（避免后续无意把张量放到 GPU）
        torch.set_default_dtype(default_dtype)  # 恢复默认 dtype

        if self.world_size > 1:  # 多进程/多 GPU 情况下，设置共享内存通道
            if rank == 0:  # 主进程：创建共享内存，通知从进程
                self.shm = SharedMemory(
                    name="nanovllm", create=True, size=2**20
                )  # 1 MiB 共享缓冲：前4字节存长度，后面存数据
                dist.barrier()  # 同步，确保从进程在访问共享内存前主进程已创建
            else:
                dist.barrier()  # 等待主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm")  # 以同名句柄连接共享内存
                self.loop()  # 从进程进入事件循环：等待主进程发来的调用指令

    def exit(self):
        if self.world_size > 1:
            self.shm.close()  # 关闭共享内存句柄（当前进程端）
            dist.barrier()  # 等待所有进程都关闭后再由 rank0 负责 unlink
            if self.rank == 0:
                self.shm.unlink()  # 主进程删除共享内存对象（系统级资源释放）
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 释放 CUDA Graph 相关缓存，防止显存泄漏
        torch.cuda.synchronize()  # 同步 GPU：确保所有内核完成
        dist.destroy_process_group()  # 销毁分布式进程组

    def loop(self):
        while True:  # 从进程的主循环：不断读取共享内存中的“方法调用”指令
            method_name, args = (
                self.read_shm()
            )  # 读取（阻塞等待 Event），反序列化得到方法名和参数
            self.call(method_name, *args)  # 在本进程实例上执行该方法
            if method_name == "exit":  # 若主进程发来 exit 指令，则退出循环
                break

    def read_shm(self):
        assert (
            self.world_size > 1 and self.rank
        )  # 仅允许在从进程调用（world_size>1 且 rank!=0）
        self.event.wait()  # 等待主进程通过 Event 通知“有新指令”
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 前 4 字节是数据长度（小端）
        method_name, *args = pickle.loads(
            self.shm.buf[4 : n + 4]
        )  # 反序列化载荷：[method_name, *args]
        self.event.clear()  # 清除事件，准备下一次等待
        return method_name, args  # 返回方法名与参数列表

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank  # 仅允许在主进程（rank==0）调用
        data = pickle.dumps([method_name, *args])  # 将调用信息序列化
        n = len(data)  # 计算字节长度
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 写入长度（前 4 字节）
        self.shm.buf[4 : n + 4] = data  # 写入载荷（从第 5 字节开始）
        for event in self.event:  # 主进程持有每个从进程对应的 Event
            event.set()  # 逐个唤醒从进程去读取共享内存

    def call(self, method_name, *args):
        if (
            self.world_size > 1 and self.rank == 0
        ):  # 在主进程：先把调用广播到从进程（共享内存+Event）
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)  # 取到当前实例的方法引用
        return method(*args)  # 本进程本地也执行（保证各 rank 行为一致）

    def warmup_model(self):
        torch.cuda.empty_cache()  # 尽量清理可释放的显存碎片
        torch.cuda.reset_peak_memory_stats()  # 重置峰值显存统计（便于后续估算可用内存）


        # 读取配置中的两项：单批最大 token 数、模型最大序列长度
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )

        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        # 使用一个“极限大”的虚拟 batch：尽量让总 token 数接近上限，从而在预热时覆盖最重负载
        # 这里把能容纳的序列数控制在 max_num_seqs 内

        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 构造 num_seqs 个全 0 token 的虚拟序列（长度 = max_model_len）

        self.run(
            seqs, True
        )  # 以 prefill 模式跑一遍，触发 Kernel 编译/缓存、工作空间分配
        torch.cuda.empty_cache()  # 再清一次缓存，回收临时占用的显存

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()  # 查询当前设备的（空闲、总）显存字节数
        used = total - free  # 已用显存
        peak = torch.cuda.memory_stats()[
            "allocated_bytes.all.peak"
        ]  # 自进程启动以来记录的峰值占用
        current = torch.cuda.memory_stats()[
            "allocated_bytes.all.current"
        ]  # 当前占用（与 used 口径不同：仅统计 PyTorch 分配）

        num_kv_heads = hf_config.num_key_value_heads // self.world_size # TP 切分后，每卡的 KV 头数
        
        # 计算每个“KV cache 块”的字节数：
        # 2（K和V）× 层数 × 每块 token 容量 × 每卡 KV 头数 × head_dim × 每元素字节数

        # head_dim = hf_config.head_dim if hasattr(hf_config, "head_dim") else hf_config.hidden_size // hf_config.num_attention_heads

        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        # block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize

        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current // block_bytes)
        # 估算可用于 KV 的显存预算：总显存 × 利用率 -（已用 + 峰值 - 当前）
        # 这样在考虑碎片和峰值后，尽量保守地给 KV cache 预留空间
        assert config.num_kvcache_blocks > 0  # 至少要能分到 1 个块，否则无法运行

        # 预分配整块 KV 张量（在默认 device=CUDA 下）：形状为 [K/V, 层, 块, 块内位置, 头, 头维]
        self.kv_cache = torch.zeros(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
        )

        layer_id = 0
        for module in self.model.modules():  # 遍历模型所有子模块
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[
                    0, layer_id
                ]  # 将对应层的视图绑定到模块属性，供前向时直接写入
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(
            len(seq.block_table) for seq in seqs
        )  # 找出本批次中 block_table 的最大长度（对齐用）
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        # 将每个序列的 block_table 右侧用 -1 填充到相同长度（-1 表示无效块）

        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        # 组装为 int32 张量；pin_memory=True 便于异步 DMA 传输到 GPU；non_blocking=True 配合异步拷贝

        return block_tables

    # Prefill：处理输入提示词阶段，需要计算所有token的注意力
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []  # 累计本批次所有需要“新计算”的 token id（忽略已缓存部分）
        positions = []  # 对应的绝对位置（从 0 开始）
        cu_seqlens_q = [0]  # 变长序列前缀和（query），用于 FlashAttention/Fused kernel
        cu_seqlens_k = [0]  # 变长序列前缀和（key）
        max_seqlen_q = 0  # 本批 query 的最大长度（未缓存的新 token 数）
        max_seqlen_k = 0  # 本批 key 的最大长度（完整上下文长度）
        slot_mapping = []  # 将“将要写入的 KV 槽位”的全局索引列表（跨块展开）
        block_tables = (
            None  # 可选：如果存在 prefix cache，需要把 block_table 传给 kernel
        )

        for seq in seqs:
            seqlen = len(seq)  # 序列当前总长度（包含已缓存+未缓存）
            input_ids.extend(seq[seq.num_cached_tokens :])  # 只追加“未缓存”的 token（prefill 新算部分）
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))  # 这些 token 的位置索引

            seqlen_q = seqlen - seq.num_cached_tokens  # 本序列需要新计算的 token 数
            seqlen_k = seqlen  # 本序列的 key 长度（全上下文）

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:  # 没有分配过 KV 块则跳过映射填充
                continue

            for i in range(seq.num_cached_blocks, seq.num_blocks):  # 仅为“新写入”的块建立槽位映射
                start = (seq.block_table[i] * self.block_size)  # 该块在全局 KV 空间的起始位置
                if i != seq.num_blocks - 1:
                    end = start + self.block_size  # 非最后一块：写满整个块
                else:
                    end = start + seq.last_block_num_tokens  # 最后一块：只写已占用的 token 数

                slot_mapping.extend(list(range(start, end)))  # 把该块内每个位置的全局索引追加进去

        if (cu_seqlens_k[-1] > cu_seqlens_q[-1]):  # prefix cache：存在已缓存的前缀（说明有些 token 不需要重算）
            block_tables = self.prepare_block_tables(seqs)  # 传入 block_tables 以便 kernel 读取历史 KV

        # 把所有准备好的 Python 列表搬到 GPU，设置合适 dtype，并使用 pinned memory + 异步拷贝
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        # 将“本批 prefill 的上下文信息”注册到线程上下文，供模型前向时的 fused kernel/attention 使用
        # 参数含义：is_prefill=True、变长前缀和、最大长度、KV 写入槽位映射、（decode 专用传 None）、以及可选的 block_tables

        return input_ids, positions  # 返回给 run_model 使用

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []  # decode 阶段：仅输入“上一个生成的 token”
        positions = []  # 该 token 的绝对位置（当前序列长度）
        slot_mapping = []  # 本次写入的 KV 槽位（与输入 token 对齐）
        context_lens = []  # 每条序列的上下文长度（供 kernel 计算注意力范围）

        for seq in seqs:
            input_ids.append(seq.last_token)  # 上一步生成的最后一个 token 作为本步输入
            positions.append(len(seq))  # 其对应的绝对位置（0-based），即当前上下文长度
            context_lens.append(len(seq))  # 同上，提供给 kernel
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            # 将要写/读的 KV 槽位：位于最后一个已分配块中，索引是“该块已用 token 数 - 1”
            # （decode 时对“当前输入 token”的 KV 进行读/写；这里映射到其在全局 KV 空间中的线性地址）

        # 拷贝到 GPU（与 prefill 相同的 pinned memory + non_blocking 策略）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        block_tables = self.prepare_block_tables(
            seqs
        )  # decode 也需要提供 block_tables 给 kernel
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        # 注册“decode 模式”的上下文信息（is_prefill=False）；此时无需 cu_seqlens/max_seqlen

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []  # 收集每条序列的温度（可不同序列不同温度）

        for seq in seqs:
            temperatures.append(seq.temperature)

        # 组装为 GPU 上的 float32 张量（小张量，拷贝成本很低）
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)

        return temperatures

    @torch.inference_mode()  # 禁用 autograd，减少显存、提高速度
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        # 三种情况走常规前向：prefill（变长/大算子）、强制 eager、或 batch>512（未捕获相应 Graph）
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
            # 前向得到隐状态，再经 compute_logits（通常是一层输出投影）得到 logits

        else:  # Decode
            bs = input_ids.size(0)  # 当前 batch size
            context = get_context()  # 取到上面 set_context 注册的 decode 上下文
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            # 选择一个“容量 >= 实际 bs”且最小可用的 CUDA Graph（避免为每个 bs 捕获图）

            graph_vars = (
                self.graph_vars
            )  # 事先分配的一组张量池（作为 graph 的输入/输出缓存）
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()  # 清零输入张量，避免残留数据影响（outputs 不清零以节省时间）

            # 将本次实际数据拷入预分配的 graph inputs（仅填前 bs 行）
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables

            graph.replay()  # 直接复放已捕获的 CUDA 计算图（省 kernel launch 开销）
            return self.model.compute_logits(
                graph_vars["outputs"][:bs]
            )  # 对复放得到的输出再投影成 logits

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 按阶段准备输入张量与上下文
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )

        # 仅在 rank0 准备温度（采样只在主进程做；从进程只算前向）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 执行前向，得到 logits (概率分布)
        logits = self.run_model(input_ids, positions, is_prefill)

        # 主进程：用采样器把 logits → 下一个 token_id（向量化采样支持 per-seq 温度）
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )

        reset_context()  # 清理线程局部上下文，避免脏数据影响下次
        return token_ids  # 主进程返回 token_id 列表；从进程返回 None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 仅为 bs≤512 的情况捕获 CUDA Graph
        max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size  # 上限情况下，一个序列最多需要的 KV 块数（向上取整）

        # 预分配一套最大规模的“图输入/输出”张量池（默认 device=CUDA，此前 __init__ 已设定）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        self.graph_bs = [1, 2, 4, 8] + list(
            range(16, max_bs + 1, 16)
        )  # 选取一组常用 bs 桶（小到大）
        self.graphs = {}  # 存放各 bs 的 CUDA Graph
        self.graph_pool = None  # 第一张图的内存池，供后续图复用

        for bs in reversed(self.graph_bs):  # 由大到小捕获，便于内存池覆盖大形状
            graph = torch.cuda.CUDAGraph()  # 新建一张 CUDA 计算图

            # 为当前 bs 设置 decode 上下文（占位张量切片），确保捕获时 kernel 形状固定
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup：先跑一次，稳定算法选择/工作区
            with torch.cuda.graph(graph, self.graph_pool):  # 进入捕获区（复用上一次的内存池以减少显存）
                outputs[:bs] = self.model(
                    input_ids[:bs], positions[:bs]
                )  # capture：记录算子图及内存分配

            if self.graph_pool is None:
                self.graph_pool = graph.pool()  # 记住第一张图的内存池，供后续图共享

            self.graphs[bs] = graph  # 保存当前 bs 的图
            torch.cuda.synchronize()  # 同步，确保捕获完成
            reset_context()  # 清理上下文，避免影响下次捕获

        self.graph_vars = dict(  # 保存这组“可复用变量”的引用，run_model 会写入它们
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
