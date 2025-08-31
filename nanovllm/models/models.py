from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM
from .llama import LlamaForCausalLM

# Register Model
model_dict = {
    "qwen3": Qwen3ForCausalLM,
    "llama": LlamaForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}