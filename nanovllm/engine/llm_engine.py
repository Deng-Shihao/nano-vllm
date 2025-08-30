import atexit  # 标准库：注册程序退出时要执行的清理回调（防止遗留子进程/资源未释放）

from dataclasses import (
    fields,
)  # 从 dataclasses 导入 fields，用于枚举数据类 Config 的字段（名字、类型等）
from time import perf_counter  # 高精度计时器，适合做吞吐/耗时统计
from tqdm.auto import tqdm  # 自动选择合适前端的进度条（Jupyter/终端都能显示）
from transformers import (
    AutoTokenizer,
)  # HuggingFace 的分词器工厂，根据模型名称自动选择合适的 tokenizer
import torch.multiprocessing as mp  # PyTorch 的多进程工具（与 CUDA 更兼容），支持 'spawn' 等启动方式

from nanovllm.config import Config  # 项目内：引擎/模型的配置数据类
from nanovllm.sampling_params import (
    SamplingParams,
)  # 采样参数（温度、top_k、max_tokens 等）
from nanovllm.engine.sequence import Sequence  # 表示一次生成任务/序列的对象
from nanovllm.engine.scheduler import (
    Scheduler,
)  # 负责把一批序列分配到模型运行阶段（prefill/decode）的调度器
from nanovllm.engine.model_runner import (
    ModelRunner,
)  # 实际调用模型前向、管理 KV 缓存等的执行器（可在多进程中运行）


class LLMEngine:
    # 一个高层引擎，封装：多进程并行（tensor parallel）、调度器、tokenizer 与推理循环

    def __init__(self, model, **kwargs):
        # 初始化时：构造 Config、创建并行子进程、初始化主进程的 ModelRunner/Tokenizer/Scheduler，并注册退出清理

        config_fields = {field.name for field in fields(Config)}
        # 取出 Config 数据类中所有字段名（用集合便于过滤 kwargs 中无关键）

        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 只保留 kwargs 里与 Config 字段匹配的键值，避免把未知参数传进 Config

        config = Config(model, **config_kwargs)
        # 构造配置对象：至少包含模型名称/权重路径，以及（可能）tensor_parallel_size 等并行参数

        self.ps = []
        # 保存子进程对象的列表（非 rank0 的并行分片进程都会放进来）

        self.events = []
        # 保存与子进程通信/同步用的事件（或事件列表），供主/子进程间协调

        ctx = mp.get_context("spawn")
        # 获取多进程上下文，使用 'spawn' 启动方式（更安全、跨平台；CUDA 场景通常推荐，不共享父进程状态）

        for i in range(1, config.tensor_parallel_size):
            # 为除 rank0 外的每个 tensor parallel 分片创建一个子进程
            event = ctx.Event()
            # 为该子进程创建一个事件对象，用来通知子进程已就绪/需要执行等

            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            # 创建子进程：目标可调用是 ModelRunner（注意：这里把类作为 target，说明其 __call__/__init__ 里会进入循环）
            # 传参包含：相同的配置对象、该进程的分片 rank（i），以及事件对象

            process.start()
            # 启动子进程，子进程会各自初始化模型分片、等待主进程调度

            self.ps.append(process)
            # 记录子进程句柄，便于退出时 join

            self.events.append(event)
            # 记录对应事件，主进程可批量把事件传给 rank0 的 ModelRunner，用于跨进程同步

        self.model_runner = ModelRunner(config, 0, self.events)
        # 在主进程创建 rank0 的 ModelRunner；把所有子分片的事件传入，rank0 负责驱动/汇总

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 根据模型名称/路径加载对应的分词器；use_fast=True 用 Rust 实现更快

        config.eos = self.tokenizer.eos_token_id
        # 把 tokenizer 的 EOS id 写回到配置，后续生成/停止条件需要

        self.scheduler = Scheduler(config)
        # 创建调度器：管理 Sequence 的生命周期、决定每步送入模型的批次，以及处理 prefill/decode 阶段

        atexit.register(self.exit)
        # 注册退出钩子：当解释器退出时自动调用 self.exit()，确保子进程被正确终止

    def exit(self):
        # 清理资源：通知模型运行器退出，并等待所有子进程结束

        self.model_runner.call("exit")
        # 通过内部 RPC/消息机制调用 ModelRunner 的 "exit" 命令，让其释放显存/结束循环

        del self.model_runner
        # 显式删除引用，帮助尽快释放相关资源

        for p in self.ps:
            p.join()
            # 等待所有子进程退出，避免僵尸进程

    # 向引擎提交一个生成请求：prompt 可是原始字符串或已编码的 token id 列表
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):

        # 如果是字符串，先用 tokenizer 编码为 token id 列表（通常不含特殊 tokens 以免重复）
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # 把 prompt 与采样参数包装成一个 Sequence（跟踪状态、KV 缓存句柄、已生成 token 等）
        seq = Sequence(prompt, sampling_params)

        # 把该序列交给调度器排队（等待被分配到 prefill 或 decode）
        self.scheduler.add(seq)

    # 执行一次调度与模型前向：跑 prefill 或 decode 的一个“步长”，并回收模型输出
    # - prefill：对提示 token 进行一次性前向，初始化 KV 缓存
    # - decode：每步生成下一个 token（通常自回归逐步进行）
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # 让调度器选择本步要处理的序列批次，以及当前是否处在 prefill 阶段

        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 驱动模型执行：把批次序列与阶段标记传给 ModelRunner，得到这些序列本步输出的 token id（decode 阶段）

        self.scheduler.postprocess(seqs, token_ids)
        # 把模型输出交回调度器进行后处理：更新每个序列的完成状态、追加 token、检查停止条件（eos/max_tokens 等）

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 收集本步“已完成”的序列：返回它们的 id 及完整生成 token 列表（不含 prompt）

        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        # 为吞吐统计记录“token 数”：
        # - prefill 阶段：以本步处理的提示 token 总数衡量吞吐（正数）
        # - decode 阶段：以本步并行生成的序列数衡量（每个序列通常只解出 1 个 token，取负号表示 decode）

        return outputs, num_tokens
        # 返回：本步完成的输出（可能为空）以及吞吐计数符号化数据

    def is_finished(self):
        return self.scheduler.is_finished()
        # 询问调度器：是否所有序列都已经完成（队列清空/全部到达停止条件）

    # 高层接口：批量生成文本，内置进度条与吞吐打印；返回每个请求的文本与 token_id
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:

        # 创建进度条：总数为请求数；dynamic_ncols 让进度条宽度自适应终端
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 如果只给了一个采样参数，则复制成与 prompts 等长的列表，所有请求共用
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 把每个请求添加到调度器队列（此时还没真正跑模型）
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 用于暂存已完成序列的 {seq_id: token_ids} 映射（seq_id 由 Sequence 内部分配，保证唯一）
        outputs = {}

        # 实时吞吐统计的两个数值：分别代表 prefill（tok/s）与 decode（tok/s）
        prefill_throughput = decode_throughput = 0.0

        # 主推理循环：直到调度器报告全部完成
        while not self.is_finished():
            # 记录 step 前时间戳，用于计算本步吞吐
            t = perf_counter()

            # 进行一步调度+前向；拿到本步完成的输出以及“吞吐计数”符号值
            output, num_tokens = self.step()

            if use_tqdm:
                if num_tokens > 0:
                    # prefill：用“处理的提示 token 数 / 耗时”估算 tok/s
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # decode：num_tokens 为负，取反代表“本步生成 token 数”（通常等于批大小），再除以耗时
                    decode_throughput = -num_tokens / (perf_counter() - t)

                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
                # 在进度条右侧展示两种阶段的实时吞吐（取整便于阅读）

            # 把本步新完成的序列结果写入字典（覆盖无所谓，因为同一 seq 只会完成一次）
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids

                # 每完成一个请求就把进度条推进一格
                if use_tqdm:
                    pbar.update(1)

        # 把字典按 seq_id 排序转成列表，确保输出顺序稳定（与提交顺序一致）
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]

        # 把每个结果解码成字符串，并保留原始 token_ids，返回结构为列表[{"text":..., "token_ids":[...]}]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]

        # 收尾：关闭进度条（释放控制台资源，避免输出错位）
        if use_tqdm:
            pbar.close()

        return (
            outputs  # 返回最终结果列表；调用者可取 outputs[i]["text"] 获得对应生成文本
        )
