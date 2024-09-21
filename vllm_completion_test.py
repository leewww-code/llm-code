from openai import OpenAI
client =OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="123456",
)
print("服务连接成功")
completion=client.completions.create(
    model="/mnt/workspace/Meta-Llama-3.1-8B",
    prompt="San Francisco is a",
    max_tokens=128,
)
print("### San Francisco is : ")
print("Completion result: ",completion)