import torch  # 导入 PyTorch 主库，提供张量/自动求导/GPU 加速等
from torch import nn  # 从 torch 中导入神经网络模块基类与常用层
import torch.distributed as dist  # 分布式训练/推理接口（NCCL/Gloo 等后端）
from transformers import (
    Qwen3Config,
)  # 来自 HuggingFace 的 Qwen3 配置对象（模型结构超参）

from nanovllm.layers.activation import (
    SiluAndMul,
)  # 自定义激活：SiLU(x) * x（SwiGLU 变体的一半逻辑）
from nanovllm.layers.attention import (
    Attention,
)  # 自定义注意力核（封装 Flash-Attn/KV cache 读写）
from nanovllm.layers.layernorm import (
    RMSNorm,
)  # 自定义 RMSNorm（均方根归一化，常用于 LLM）
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

# 上面三者是“张量并行”的线性层实现：
# - QKVParallelLinear：一次性并行计算 Q/K/V，并按张量并行切分
# - MergedColumnParallelLinear：列并行线性层，支持多个投影合并（如 gate/up 合并）
# - RowParallelLinear：行并行线性层，常用于输出投影（需要全卡 All-Reduce）

from nanovllm.layers.rotary_embedding import (
    get_rope,
)  # 构造 RoPE（旋转位置编码）算子/函数
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

# 词嵌入和 LM 头的张量并行版本（词表维度分片）


class Qwen3Attention(
    nn.Module
):  # 单层自注意力模块（含 QKV 投影、RoPE、注意力核与输出投影）

    def __init__(
        self,
        hidden_size: int,  # 模型隐藏维度
        num_heads: int,  # 总注意力头数（未切分前）
        num_kv_heads: int,  # K/V 头数（多查询注意力 MQA/MKV 结构）
        max_position: int = 4096 * 32,  # RoPE 最大支持位置（用于长上下文）
        head_dim: (
            int | None
        ) = None,  # 每个头的维度；若 None 则用 hidden_size // num_heads
        rms_norm_eps: float = 1e-06,  # RMSNorm 的数值稳定项
        qkv_bias: bool = False,  # QKV 线性层是否使用 bias
        rope_theta: float = 10000,  # RoPE 的频率基数（θ）
        rope_scaling: tuple | None = None,  # RoPE 的缩放策略（用于扩展上下文）
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()  # 当前张量并行进程/设备数
        self.total_num_heads = num_heads  # 记录总头数
        assert self.total_num_heads % tp_size == 0  # 头数必须能被 TP 规模整除
        self.num_heads = (
            self.total_num_heads // tp_size
        )  # 本卡负责的注意力头数（切分后）

        self.total_num_kv_heads = num_kv_heads  # 总的 K/V 头数
        assert self.total_num_kv_heads % tp_size == 0  # 同样要求能整除
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 本卡的 K/V 头数

        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 推导每头维度
        self.q_size = self.num_heads * self.head_dim  # 本卡 Q 向量拼接后的通道数
        self.kv_size = self.num_kv_heads * self.head_dim  # 本卡 K/V 各自的通道数
        self.scaling = self.head_dim**-0.5  # 点积注意力缩放因子 1/sqrt(d)

        self.qkv_proj = (
            QKVParallelLinear(  # 张量并行的 QKV 合并线性层：一次 matmul 出 Q/K/V
                hidden_size,  # 输入通道 = hidden_size
                self.head_dim,  # 单头维度
                self.total_num_heads,  # 总 Q 头数（由层内部负责切分）
                self.total_num_kv_heads,  # 总 KV 头数
                bias=qkv_bias,  # 是否带偏置
            )
        )
        self.o_proj = RowParallelLinear(  # 行并行的输出投影（合并各头的输出）
            self.total_num_heads * self.head_dim,  # 输入通道 = 所有头拼接后的维度
            hidden_size,  # 输出回到隐藏维度
            bias=False,
        )
        self.rotary_emb = get_rope(  # 构建 RoPE 编码器
            self.head_dim,  # 旋转作用的维度（一般等于 head_dim）
            rotary_dim=self.head_dim,  # 指定对多少维应用 RoPE（通常=head_dim）
            max_position=max_position,  # 支持的最大相对/绝对位置
            base=rope_theta,  # RoPE 频率基数
            rope_scaling=rope_scaling,  # 可选的长距缩放策略
        )
        self.attn = Attention(  # 注意力算子（可包融合 kernel / KV cache 访问）
            self.num_heads,  # 本卡 Q 头数
            self.head_dim,  # 单头维度
            self.scaling,  # 缩放因子
            self.num_kv_heads,  # 本卡 KV 头数（MQA/MKV 支持）
        )

        self.q_norm = RMSNorm(
            self.head_dim, eps=rms_norm_eps
        )  # Q 上的 RMSNorm（Qwen3 特有）
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)  # K 上的 RMSNorm

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引（RoPE 需要）
        hidden_states: torch.Tensor,  # 输入隐状态 [B*T, H] 或 [N, H]（已被外部展平/拼批）
    ) -> torch.Tensor:
        qkv = self.qkv_proj(
            hidden_states
        )  # 线性得到串联的 [Q|K|V]（张量并行内部已分片）
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 按本卡尺寸切出 Q/K/V 张量

        q_by_head = q.view(
            -1, self.num_heads, self.head_dim
        )  # 将 Q 变形为 [*, num_heads, head_dim]
        q_by_head = self.q_norm(q_by_head)  # 对每个头的通道做 RMSNorm
        q = q_by_head.view(q.shape)  # 还原回原形状（与后续内核期望对齐）

        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)  # 同理处理 K
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)  # 注入 RoPE（对 Q/K 做旋转位置编码）
        o = self.attn(q, k, v)  # 调用注意力核：softmax(QK^T/√d)V（含 KV cache）
        output = self.o_proj(o)  # 多头拼接后的输出映射回隐藏维度
        return output  # 返回注意力输出（供残差/MLP 使用）


class Qwen3MLP(nn.Module):  # FFN 前馈网络模块（SwiGLU 结构）

    def __init__(
        self,
        hidden_size: int,  # 输入/输出隐藏维度
        intermediate_size: int,  # 中间扩展维度（FFN 扩张倍数）
        hidden_act: str,  # 激活函数名，Qwen3 使用 "silu"
    ) -> None:
        super().__init__()

        self.gate_up_proj = (
            MergedColumnParallelLinear(  # 合并两路投影：gate_proj 与 up_proj（列并行）
                hidden_size,  # 输入维度
                [intermediate_size] * 2,  # 输出两支各为 intermediate_size
                bias=False,
            )
        )
        self.down_proj = (
            RowParallelLinear(  # 行并行的回投影（把中间维度降回 hidden_size）
                intermediate_size,
                hidden_size,
                bias=False,
            )
        )
        assert hidden_act == "silu"  # Qwen3 采用 SiLU
        self.act_fn = SiluAndMul()  # 实现 SwiGLU：SiLU(gate) * up

    def forward(self, x):
        gate_up = self.gate_up_proj(
            x
        )  # 计算 [gate, up] 合并的输出，形状 [..., 2*intermediate]
        x = self.act_fn(gate_up)  # 拆分为 gate/up，做 SiLU(gate) * up
        x = self.down_proj(x)  # 再映射回 hidden_size
        return x  # 返回 FFN 输出


class Qwen3DecoderLayer(
    nn.Module
):  # 解码器层：RMSNorm → Self-Attn → RMSNorm → MLP（带残差）

    def __init__(
        self,
        config: Qwen3Config,  # HF 的配置对象（包含所有层超参）
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(  # 构建注意力子层
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(  # 构建 FFN 子层（SwiGLU）
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 前置归一化（Pre-Norm）：在进入注意力前做 RMSNorm

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # 注意力后的第二个 RMSNorm（常与残差融合实现）

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引（传给注意力以施加 RoPE）
        hidden_states: torch.Tensor,  # 输入隐状态
        residual: torch.Tensor | None,  # 残差分支；None 表示本层开头尚未构建残差
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if residual is None:  # 第一子层入口：初始化残差
            residual = hidden_states  # 把原输入作为残差
            hidden_states = self.input_layernorm(hidden_states)  # 对主分支做 RMSNorm
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )  # 支持 fused 的 RMSNorm 接口：同时处理主分支与残差（实现细节由 RMSNorm 决定）

        hidden_states = self.self_attn(positions, hidden_states)  # 自注意力
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # 第二次 RMSNorm（同样可能是 fused：返回新的主分支与残差）

        hidden_states = self.mlp(hidden_states)  # 通过 FFN
        return hidden_states, residual  # 返回主分支与残差信息（供下一子层继续用）


class Qwen3Model(nn.Module):  # 仅包含嵌入 + 多层解码器 + 最终 Norm 的主体模型
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        # 词嵌入（张量并行版本，按词表维度切分到多卡）

        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # 堆叠 N 层解码器层

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 最后再做一次 RMSNorm（Pre-Norm 架构的末尾规范化）

    def forward(
        self,
        input_ids: torch.Tensor,  # token id（通常展平后的）形状 [N] 或 [B, T]
        positions: torch.Tensor,  # 位置索引（与 input_ids 对齐）
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)  # 词嵌入 → 隐状态
        residual = None  # 初始化残差指针
        for layer in self.layers:  # 逐层通过解码器
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(
            hidden_states, residual
        )  # 末尾 RMSNorm（可能 fused 残差）
        return hidden_states  # 返回最终的隐状态序列（未投影到词表）


class Qwen3ForCausalLM(nn.Module):  # 带语言模型头（lm_head）的封装：hidden → logits
    packed_modules_mapping = {
        "q_proj": (
            "qkv_proj",
            "q",
        ),  # 权重加载时的别名映射：将 HuggingFace 权重名映射到本实现的合并层切片
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)  # 主体 Transformer（不含 lm_head）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 词表并行的输出头：把隐藏向量映射成词表 logits（按词表切分，最终需要 All-Reduce/Concat）

        if (
            config.tie_word_embeddings
        ):  # 参数共享：输入嵌入与输出权重共享（weight tying）
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,  # token id
        positions: torch.Tensor,  # 位置索引
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)  # 先得到隐状态
        return hidden_states  # 不在这里算 logits（给灵活调用者选择）

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # 来自 forward 的隐状态
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)  # 经过并行 LM 头得到词表 logits
        return logits  # 返回最终 logits（供采样/损失计算等）
