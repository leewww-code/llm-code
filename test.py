import transformers
import torch

pipeline = transformers.pipeline(
    task="text-generation",
    model="/mnt/workspace/Meta-Llama-3.1-8B",
    model_kwargs={"torch_dtype":torch.bfloat16},
    device="cuda",
)
print(pipeline("Hey how are you doing today ?"))
