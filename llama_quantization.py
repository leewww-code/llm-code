from vllm import LLM, SamplingParams
import torch

torch.cuda.empty_cache()

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=64)
# device = torch.device('cuda:2')
llm = LLM(
    model="/mnt/workspace/Meta-Llama-3.1-8B", 
    #quantization='awq',
    gpu_memory_utilization=0.8,
    trust_remote_code=True 
)
# llm.to(device)
prompts = [
    #"Hello, my name is",
    "The president of the United States is",
    #"The capital of France is",
    "The future of AI is",
    #"One way to crack a password",
    #"I know unsensored swear words such as"
]
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
