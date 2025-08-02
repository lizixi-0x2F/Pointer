import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

# 直接导入模型
from src.model.pointer_model import PointerDecoder
from benchmarks.baselines import MambaBaseline, PerformerBaseline, TransformerBaseline
from tqdm import tqdm
import datasets
import argparse

def load_dataset(dataset_name):
    """加载listops-32或wikitext-2数据集"""
    if dataset_name == "listops-32":
        # 从本地路径加载listops数据集
        local_path = "/Volumes/oz/pointer/listops-32"
        train = datasets.load_dataset(local_path, split="train")
        val = datasets.load_dataset(local_path, split="validation")
        test = datasets.load_dataset(local_path, split="test")
        
        # 简单预处理
        vocab_size = 32
        max_len = 2048
        
        def preprocess(batch):
            # 检查并适配不同字段名
            tokens = batch.get("tokens") or batch.get("input_ids") or batch["input_tokens"]
            label = batch.get("label") or batch.get("labels") or 0  # 默认0如果无标签
            
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long)
            }
            
    elif dataset_name == "wikitext-2":
        train = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        val = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        test = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # 简单预处理
        vocab_size = 50257  # GPT-2词汇表大小
        max_len = 1024
        
        def preprocess(batch):
            return {
                "input_ids": torch.tensor(batch["text"], dtype=torch.long),
                "labels": torch.tensor(batch["text"], dtype=torch.long)  # 语言建模任务
            }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train, val, test, vocab_size, max_len, preprocess

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """训练模型并评估"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(inputs)
            logits = outputs["logits"]
            
            if logits.dim() == 3:  # 语言建模任务
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(inputs)
                logits = outputs["logits"]
                
                if logits.dim() == 3:
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {100.*val_correct/val_total:.2f}%")

def run_benchmark(dataset_name, model_type="pointer", epochs=10):
    """运行基准测试"""
    # 加载数据
    train, val, test, vocab_size, max_len, preprocess = load_dataset(dataset_name)
    
    # 创建数据加载器
    train_loader = DataLoader(train.map(preprocess), batch_size=16, shuffle=True)
    val_loader = DataLoader(val.map(preprocess), batch_size=16)
    test_loader = DataLoader(test.map(preprocess), batch_size=16)
    
    # 初始化模型
    if model_type == "pointer":
        model = PointerDecoder(
            vocab_size=vocab_size,
            d=128,
            n_layers=4,
            n_heads=4,
            max_seq_len=max_len
        )
    elif model_type == "mamba":
        model = MambaBaseline(vocab_size=vocab_size, d_model=128, n_layers=4)
    elif model_type == "performer":
        model = PerformerBaseline(vocab_size=vocab_size, d_model=128, n_layers=4)
    elif model_type == "transformer":
        model = TransformerBaseline(vocab_size=vocab_size, d_model=128, n_layers=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 训练和评估
    print(f"\n=== Benchmarking {model_type} on {dataset_name} ===")
    train_model(model, train_loader, val_loader, epochs=epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="listops-32", 
                        choices=["listops-32", "wikitext-2"])
    parser.add_argument("--model", type=str, default="pointer",
                        choices=["pointer", "mamba", "performer", "transformer"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    run_benchmark(args.dataset, args.model, args.epochs)
