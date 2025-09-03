import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen2Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm

from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from nanovllm.layers.rotary_embedding import get_rope


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()  # 当前张量并行进程/设备数

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # 张量并行的 QKV 合并线性层：一次 matmul 出 Q/K/V
        self.qkv_proj = QKVParallelLinear(
            hidden_size,  # 输入通道 = hidden_size
            self.head_dim,  # 单头维度
            self.total_num_heads,  # 总 Q 头数（由层内部负责切分）
            self.total_num_kv_heads,  # 总 KV 头数
            bias=qkv_bias,  # 是否带偏置
        )
        # 行并行的输出投影（合并各头的输出
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,  # 输入通道 = 所有头拼接后的维度
            hidden_size,  # 输出回到隐藏维度
            bias=False,
        )
        # 构建 RoPE 编码器
        self.rotary_emb = get_rope(
            self.head_dim,  # 旋转作用的维度（一般等于 head_dim）
            rotary_dim=self.head_dim,  # 指定对多少维应用 RoPE（通常=head_dim）
            max_position=max_position,  # 支持的最大相对/绝对位置
            base=rope_theta,  # RoPE 频率基数
            rope_scaling=rope_scaling,  # 可选的长距缩放策略
        )
        # 注意力算子（可包融合 kernel / KV cache 访问）
        self.attn = Attention(
            self.num_heads,  # 本卡 Q 头数
            self.head_dim,  # 单头维度
            self.scaling,  # 缩放因子
            self.num_kv_heads,  # 本卡 KV 头数（MQA/MKV 支持）
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引（RoPE 需要）
        hidden_states: torch.Tensor,  # 输入隐状态 [B*T, H] 或 [N, H]（已被外部展平/拼批）
    ) -> torch.Tensor:
        qkv = self.qkv_proj(
            hidden_states
        )  # 线性得到串联的 [Q|K|V]（张量并行内部已分片）
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )  # 按本卡尺寸切出 Q/K/V 张量
        q, k = self.rotary_emb(positions, q, k)  # 注入 RoPE（对 Q/K 做旋转位置编码）
        o = self.attn(q, k, v)  # 调用注意力核：softmax(QK^T/√d)V（含 KV cache）
        output = self.o_proj(o)  # 多头拼接后的输出映射回隐藏维度
        return output  # 返回注意力输出（供残差/MLP 使用）


# FFN 前馈网络模块（SwiGLU 结构）
class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,  # 输入/输出隐藏维度
        intermediate_size: int,  # 中间扩展维度（FFN 扩张倍数）
        hidden_act: str,  # 激活函数名，Qwen2 使用 "silu"
    ) -> None:
        super().__init__()
        # 合并两路投影：gate_proj 与 up_pr# FFN 前馈网络模块（SwiGLU 结构）oj（列并行）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出两支各为 intermediate_size
            bias=False,
        )

        # 合并两路投影：gate_proj 与 up_pr# FFN 前馈网络模块（SwiGLU 结构）oj（列并行）
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        assert hidden_act == "silu"  # Qwen2 采用 SiLU
        self.act_fn = SiluAndMul()  # 实现 SwiGLU：SiLU(gate) * up

    def forward(self, x):
        gate_up = self.gate_up_proj(
            x
        )  # 计算 [gate, up] 合并的输出，形状 [..., 2*intermediate]
        x = self.act_fn(gate_up)  # 拆分为 gate/up，做 SiLU(gate) * up
        x = self.down_proj(x)  # 再映射回 hidden_size
        return x  # 返回 FFN 输出


# 解码器层：RMSNorm → Self-Attn → RMSNorm → MLP（带残差）
class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,  # HF 的配置对象（包含所有层超参）
    ) -> None:
        super().__init__()
        # 合并两路投影：gate_proj 与 up_pr# FFN 前馈网络模块（SwiGLU 结构）oj（列并行）
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # 构建 FFN 子层（SwiGLU）
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # 前置归一化（Pre-Norm）：在进入注意力前做 RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 注意力后的第二个 RMSNorm（常与残差融合实现）
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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


class Qwen2Model(nn.Module):  # 仅包含嵌入 + 多层解码器 + 最终 Norm 的主体模型
    def __init__(
        self,
        config: Qwen2Config,
    ) -> None:
        super().__init__()

        # 词嵌入（张量并行版本，按词表维度切分到多卡）
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        # 堆叠 N 层解码器层
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # 最后再做一次 RMSNorm（Pre-Norm 架构的末尾规范化）
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class Qwen2ForCausalLM(nn.Module):  # 带语言模型头（lm_head）的封装：hidden → logits
    # 词嵌入（张量并行版本，按词表维度切分到多卡）
    packed_modules_mapping = {
        "q_proj": (
            "qkv_proj",
            "q",
        ),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)  # 主体 Transformer（不含 lm_head）

        # 词表并行的输出头：把隐藏向量映射成词表 logits（按词表切分，最终需要 All-Reduce/Concat）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 词嵌入（张量并行版本，按词表维度切分到多卡）
        if config.tie_word_embeddings:
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
