import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 保证 logits 是 float 类型
        logits = logits.float()

        # 贪心解码（直接取最大值 token）
        greedy_tokens = logits.argmax(dim=-1)

        # 温度缩放
        logits.div_(temperatures.unsqueeze(dim=1))

        # softmax 得到概率分布
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)

        # Gumbel-Max trick：用指数噪声做采样
        epsilon = 1e-10
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1) + epsilon
        ).argmax(dim=-1)

        # 如果温度为 0 → 用贪心解码，否则 → 随机采样
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)


# ================= 测试 =================
if __name__ == "__main__":
    sampler = Sampler()

    # 模拟 logits（batch_size=2, vocab_size=5）
    logits = torch.tensor(
        [
            [1.0, 2.0, 0.5, 0.1, -1.0],  # 第一个样本
            [0.2, 0.1, 3.0, 0.5, 1.5],  # 第二个样本
        ]
    )

    # 定义不同的温度（一个用贪心，一个用采样）
    temperatures = torch.tensor([0.0, 1.0])  # 第一个样本=贪心，第二个样本=采样

    # 执行采样
    tokens = sampler(logits, temperatures)

    print("输入 logits:")
    print(logits)
    print("\n温度:", temperatures.tolist())
    print("输出 token id:", tokens.tolist())
