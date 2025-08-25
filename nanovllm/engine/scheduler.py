from collections import deque

# deque：双端队列，比 list 更高效地在头尾插入/弹出。
# 这里用于实现“等待队列”和“运行队列”，方便调度任务。

from nanovllm.config import Config

# 引擎配置类，包含调度器所需的参数，比如 max_num_seqs、max_num_batched_tokens 等。

from nanovllm.engine.sequence import Sequence, SequenceStatus

# Sequence：表示一个请求（包含提示、采样参数、状态、缓存的 token 等）
# SequenceStatus：枚举，可能是 WAITING、RUNNING、FINISHED，用来标记序列状态。

from nanovllm.engine.block_manager import BlockManager

# BlockManager：管理 KV cache 内存的分配和释放。
# 每个序列需要占用一定数量的 KV cache block，调度时必须确保内存够用。


# 调度器：管理等待队列和运行队列，负责决定哪些序列进入模型运行
# 支持 prefill（提示阶段）和 decode（逐 token 生成阶段）两种模式。
class Scheduler:
    def __init__(self, config: Config):

        # 单次调度最多能同时处理多少条序列（批大小上限）
        self.max_num_seqs = config.max_num_seqs

        # 单次批处理 token 总数上限（控制 GPU 显存/计算负载）
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # 终止符的 token id，用于判断序列是否完成
        self.eos = config.eos

        # 创建一个 KV 缓存块管理器，负责显存的分配与回收。
        # 参数：块数量 & 每个块大小。
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )

        # 等待队列：新请求先进入这里，等待分配显存和调度。
        self.waiting: deque[Sequence] = deque()

        # 运行队列：已经进入 prefill/decode 的序列。
        self.running: deque[Sequence] = deque()

    # 如果两个队列都空，说明所有请求都完成了。
    def is_finished(self):
        return not self.waiting and not self.running

    # 把新请求放到等待队列尾部，等待调度。
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 返回：本次调度的序列列表，以及是否是 prefill 阶段。
        # 整体逻辑：
        # 1. 优先尝试 prefill（等待队列有新序列）
        # 2. 如果没有新序列可以 prefill，再调度 decode（运行队列中的序列）

        scheduled_seqs = []  # 记录本次被调度的序列
        num_seqs = 0  # 已经调度的序列数
        num_batched_tokens = 0  # 当前批次 token 数

        # === prefill 阶段 ===
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 查看等待队列头部的序列（但不弹出）

            # 如果当前批次 token 超过限制，或者 KV 缓存不足，就不能再添加序列。
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break

            # 为该序列分配 KV 缓存块（存储注意力 key/value）
            num_seqs += 1
            self.block_manager.allocate(seq)

            # 增加 token 计数：prefill 时需要计算未缓存的 token。
            # 对新序列来说 seq.num_cached_tokens=0，因此就是整个 prompt 长度。
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            # 标记为运行中（RUNNING）
            seq.status = SequenceStatus.RUNNING

            # 从等待队列移出
            self.waiting.popleft()

            # 加入运行队列
            self.running.append(seq)

            # 加入本次调度的序列列表
            scheduled_seqs.append(seq)

        # 如果有 prefill 序列，则立即返回，标记 True（prefill 阶段）。
        if scheduled_seqs:
            return scheduled_seqs, True

        # === decode 阶段 ===
        while self.running and num_seqs < self.max_num_seqs:
            # 从运行队列头取一个序列
            seq = self.running.popleft()

            # 如果 KV 缓存不足，无法给该序列继续追加 token
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                    # 从队尾拿一个正在运行的序列，抢占它的显存（送回等待队列）

                else:
                    self.preempt(seq)
                    # 如果没有其他序列可抢占，只能自己退回等待队列
                    break

            else:
                # 如果能追加 KV 缓存
                num_seqs += 1
                self.block_manager.may_append(seq)
                # 申请 KV 缓存块，用于存储新生成的 token
                scheduled_seqs.append(seq)

        assert scheduled_seqs
        # 确保 decode 阶段至少调度到一个序列，否则逻辑错误。

        self.running.extendleft(reversed(scheduled_seqs))
        # 把调度到的序列重新放回运行队列的左边（保持顺序）。
        # 因为 deque.extendleft 会按逆序插入，所以要先 reversed。

        return scheduled_seqs, False
        # 返回调度到的序列，标记 False（decode 阶段）。

    def preempt(self, seq: Sequence):
        # 抢占：把正在运行的序列挪回等待队列，释放资源
        seq.status = SequenceStatus.WAITING
        # 标记回 WAITING

        self.block_manager.deallocate(seq)
        # 释放其占用的 KV 缓存

        self.waiting.appendleft(seq)
        # 放回等待队列的头部（优先级最高，下次立刻尝试）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        # 在模型 forward 之后调用，处理每个序列生成的 token
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # 把模型生成的 token 附加到序列中

            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                # 如果生成了 eos 且未被忽略，或者达到了最大长度，就结束序列

                seq.status = SequenceStatus.FINISHED
                # 标记为完成

                self.block_manager.deallocate(seq)
                # 释放 KV 缓存

                self.running.remove(seq)
                # 从运行队列移除
