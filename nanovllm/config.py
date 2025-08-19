import os # 用于操作文件系统（比如检查模型目录是否存在）
from dataclasses import dataclass # Python 的语法糖，可以自动生成类的初始化函数 __init__，并支持更清晰的配置对象定义
from transformers import AutoConfig # AutoConfig: HuggingFace 提供的类，用于加载预训练模型的配置（比如隐藏层大小、最大序列长度等）


@dataclass
class Config:
    model: str # 表示模型的目录路径
    max_num_batched_tokens: int = 16384 # 每次批处理的最大 token 数量，用于限制显存开销
    max_num_seqs: int = 512 # 批次中能并行处理的最大序列数
    max_model_len: int = 4096 # 模型支持的最大序列长度（可被下方调整）
    gpu_memory_utilization: float = 0.9 # 表示允许占用多少比例的 GPU 显存（通常 <1，避免 OOM）
    tensor_parallel_size: int = 1 # 张量并行的分片数量（常见于多 GPU 场景，最大值约束在 8）
    enforce_eager: bool = False # 是否强制使用 eager mode（非 graph 优化模式，调试时可能用）
    hf_config: AutoConfig | None = None # 存储 HuggingFace 加载的模型配置对象
    eos: int = -1 # End-of-Sequence token 的 ID，默认 -1 表示未指定
    kvcache_block_size: int = 256 # KV 缓存的块大小，用于 Transformer 推理加速。必须是 256 的倍数
    num_kvcache_blocks: int = -1 # KV 缓存块的数量，-1 表示自动计算

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
