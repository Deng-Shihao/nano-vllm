import os
import argparse

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main(args):
    # 模型路径
    path = os.path.expanduser(args.model_path)

    # 从path中加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

    # llm_engine 实例化
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the example script for nano-vllm.")

    parser.add_argument("--enforce_eager", "-ee", type=bool, default=True)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1)
    parser.add_argument("--model_path", "-p", type=str, default="./Qwen3-0.6B")
    parser.add_argument("--temperature", "-t", type=float, default=0.9)
    parser.add_argument("--max_tokens", "-mt", type=int, default=256)

    args = parser.parse_args()

    main(args)
