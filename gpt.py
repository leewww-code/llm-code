import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, default_data_collator
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# 参数设置
EPOCHS = 1  # 可调节的 epoch 参数
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据集类
class BookCorpusDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        self.examples = []
        with open(file_path, "r") as f:
            for line in f:
                tokenized = tokenizer(
                    line.strip(), 
                    truncation=True, 
                    max_length=max_length, 
                    padding="max_length", 
                    return_tensors="pt"
                )
                self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"].squeeze(0),
            "attention_mask": self.examples[idx]["attention_mask"].squeeze(0)
        }

# 加载 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 使用默认配置初始化 GPT-2 模型
config = GPT2Config()  # 使用默认 GPT-2 配置
model = GPT2LMHeadModel(config)  # 从配置初始化模型
model.resize_token_embeddings(len(tokenizer))
model = model.to(DEVICE)

# 加载数据
data_path = "autodl-tmp/books_subset_10000.txt"  # 假设已生成的文本文件
dataset = BookCorpusDataset(tokenizer, data_path, max_length=MAX_SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 开始训练
batch_losses = []  # 保存每个 batch 的 loss
epoch_losses = []  # 保存每个 epoch 的平均 loss

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        # 将数据移动到 GPU
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 记录损失
        batch_losses.append(loss.item())
        epoch_loss += loss.item()

        # 更新进度条
        progress_bar.set_postfix({"loss": loss.item()})

    # 记录每个 epoch 的平均 loss
    avg_loss = epoch_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch + 1} finished with avg loss: {avg_loss:.4f}")

# 保存 batch-level 损失曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(batch_losses) + 1), batch_losses, label="Batch Loss", alpha=0.7)
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Batch-level Training Loss Curve")
plt.legend()
plt.grid()
plt.savefig("batch_loss_curve.png")
plt.show()

# 保存 epoch-level 损失曲线
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', label="Epoch Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch-level Training Loss Curve")
plt.legend()
plt.grid()
plt.savefig("epoch_loss_curve.png")
plt.show()

print("Training complete. Loss curves saved as 'batch_loss_curve.png' and 'epoch_loss_curve.png'")

