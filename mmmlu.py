import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B",
   
    device_map="balanced" ) # 使用"balanced"或"balanced_low_0"以均匀分布到两张 GPU 卡上).cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

def evaluate(data, model, tokenizer):
    correct = 0
    total = len(data)
    start_time = time.time()
#将prompt设置为prompt = f"问题: {Question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n请你给出答案（只输出一个字母）: "
    for _, row in data.iterrows():
        prompt = f"问题: {row['Question']}\nA) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\n请你给出答案（只输出一个字母）: "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=512,pad_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if prediction == row['Answer']:
            correct += 1

    accuracy = correct / total
    total_time = time.time() - start_time

    return accuracy, total_time

data = pd.read_csv("autodl-tmp/moral_scenarios_top100.csv")
accuracy, total_time = evaluate(data, model, tokenizer)
print(f"Accuracy: {accuracy}, Total Time: {total_time}")

