from copy import copy  # 浅拷贝，用于复制输入的 token_ids，避免外部修改影响内部
from enum import Enum, auto  # 定义枚举类 SequenceStatus，状态管理
from itertools import count  # 迭代器，生成自增 ID

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256  # 每个 block（缓存分块）的 token 数，默认 256
    counter = count()  # 全局计数器，用于给每个 Sequence 分配唯一的 seq_id

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)  # 唯一 ID（自增）
        self.status = SequenceStatus.WAITING  # 初始状态为 WAITING
        self.token_ids = copy(token_ids)  # 存储整个序列的 token ID
        self.last_token = token_ids[-1]  # 最后一个 token
        self.num_tokens = len(self.token_ids)  # 总 token 数
        self.num_prompt_tokens = len(
            token_ids
        )  # 提示部分的 token 数（初始就是输入长度）
        self.num_cached_tokens = 0  # 已缓存的 token 数（KV cache）
        self.block_table = []  # 用于记录分配给该序列的缓存块
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # len(sequence)，返回 token 数量
    def __len__(self):
        return self.num_tokens

    # 支持索引访问 sequence[i]，返回对应 token
    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):  # 生成的 token 数（不包括 prompt）
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):  # 提示部分 token
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):  # 生成部分 token
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):  # 缓存的 block 数
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):  # 总共需要多少个 block 存储 token
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):  # 最后一个 block 中的 token 数（不一定满 256）
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 返回第 i 个 block 的 token 切片
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    # 追加一个新 token 并更新相关信息
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 对象序列化（pickle） 的协议
    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
