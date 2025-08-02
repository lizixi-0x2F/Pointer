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
        
        def preprocess(batch, max_len=2048):
            # 检查并适配不同字段名
            if "tokens" in batch:
                tokens = batch["tokens"]
            elif "input_ids" in batch:
                tokens = batch["input_ids"]
            elif "Source" in batch:
                # 处理listops-32的特殊格式
                import re
                tokens = [int(x) for x in re.findall(r'\d+', batch["Source"])]
            else:
                raise ValueError(f"Invalid batch format. Expected 'tokens', 'input_ids' or 'Source', got: {batch.keys()}")
            
            # 确保tokens是列表且长度不超过max_len
            tokens = tokens[:max_len] if isinstance(tokens, (list, tuple)) else [tokens]
            
            # 处理标签字段
            if "label" in batch:
                label = batch["label"]
            elif "labels" in batch:
                label = batch["labels"]
            elif "Target" in batch:
                label = batch["Target"]
            else:
                label = 0  # 默认0如果无标签
            
            # 确保tokens是列表或数组
            if isinstance(tokens, (list, tuple)) or (hasattr(tokens, "__array__")):
                pass  # 有效格式
            else:
                raise ValueError(f"Invalid tokens format. Expected list/array, got: {type(tokens)}")
            
            # Padding处理
            original_len = len(tokens)
            pad_len = max_len - len(tokens)
            if pad_len > 0:
                tokens = tokens + [0] * pad_len  # 用0填充
            
            # 确保所有tensor都是固定长度
            tokens = tokens[:max_len]  # 截断到max_len
            
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
                "attention_mask": torch.tensor([1]*min(original_len, max_len) + [0]*max(0, max_len-original_len), dtype=torch.long)
            }
            
    elif dataset_name == "wikitext-2":
        # 从本地路径加载wikitext-2数据集
        local_path = "/Volumes/oz/pointer/wikitext-2"
        train = datasets.load_dataset(local_path, split="train")
        val = datasets.load_dataset(local_path, split="validation")
        test = datasets.load_dataset(local_path, split="test")
        
        # 简单预处理
        vocab_size = 50257  # GPT-2词汇表大小
        max_len = 1024
        
        def preprocess(batch, max_len=1024):
            # 检查并适配不同字段名
            if "text" in batch:
                text = batch["text"]
            elif "input_ids" in batch:
                text = batch["input_ids"]
            else:
                raise ValueError(f"Invalid batch format. Expected 'text' or 'input_ids', got: {batch.keys()}")
            
            # 确保text是列表且长度不超过max_len
            text = text[:max_len] if isinstance(text, (list, tuple)) else [text]
            
            # 确保text是列表或数组
            if isinstance(text, (list, tuple)) or (hasattr(text, "__array__")):
                pass  # 有效格式
            else:
                raise ValueError(f"Invalid text format. Expected list/array, got: {type(text)}")
            
            # Padding处理
            original_len = len(text)
            pad_len = max_len - len(text)
            if pad_len > 0:
                text = text + [0] * pad_len  # 用0填充
            
            # 确保所有tensor都是固定长度
            text = text[:max_len]  # 截断到max_len
            
            return {
                "input_ids": torch.tensor(text, dtype=torch.long),
                "labels": torch.tensor(text, dtype=torch.long),
                "attention_mask": torch.tensor([1]*min(original_len, max_len) + [0]*max(0, max_len-original_len), dtype=torch.long)
            }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train, val, test, vocab_size, max_len, preprocess

def train_model(model, train_loader, val_loader, epochs=1, lr=0.001):
    """训练模型并评估"""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
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
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            logits = outputs["logits"]
            
            # 检查任务类型：分类 vs 语言建模
            if labels.dim() == 1:  # 分类任务：labels是[B]
                # logits应该是[B, num_classes]
                if logits.dim() == 3:  # 如果是[B, N, vocab_size]，取第一个token
                    logits = logits[:, 0, :]  # [B, vocab_size]
                loss = criterion(logits, labels)
            else:  # 语言建模任务：labels是[B, N]
                if logits.dim() == 3:  # [B, N, vocab_size]
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
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                logits = outputs["logits"]
                
                # 检查任务类型：分类 vs 语言建模
                if labels.dim() == 1:  # 分类任务：labels是[B]
                    # logits应该是[B, num_classes]
                    if logits.dim() == 3:  # 如果是[B, N, vocab_size]，取第一个token
                        logits = logits[:, 0, :]  # [B, vocab_size]
                    loss = criterion(logits, labels)
                else:  # 语言建模任务：labels是[B, N]
                    if logits.dim() == 3:  # [B, N, vocab_size]
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

def run_benchmark(dataset_name, models=["pointer", "mamba", "performer", "transformer"], epochs=10):
    """运行基准测试"""
    # 加载数据
    train, val, test, vocab_size, max_len, preprocess = load_dataset(dataset_name)
    
    # 创建数据加载器 - 确保max_len传递给preprocess 
    # 使用set_format确保返回pytorch tensors
    train_processed = train.map(lambda x: preprocess(x, max_len=max_len))
    train_processed.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
    
    val_processed = val.map(lambda x: preprocess(x, max_len=max_len))
    val_processed.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
    
    test_processed = test.map(lambda x: preprocess(x, max_len=max_len))
    test_processed.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
    
    train_loader = DataLoader(
        train_processed, 
        batch_size=16, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_processed,
        batch_size=16
    )
    test_loader = DataLoader(
        test_processed,
        batch_size=16
    )
    
    results = []
    
    for model_type in models:
        # 初始化模型
        if model_type == "pointer":
            # 简化配置：移除硬编码的反思层，现在所有层都有反思能力
            reflection_config = {
                # 全局反思机制，无需指定特定层
                'global_reflection': True,
                'learnable_branching': True
            }
            
            model = PointerDecoder(
                vocab_size=vocab_size,
                d=128,
                n_layers=4,
                n_heads=4,
                max_seq_len=max_len,
                reflection_config=reflection_config  # 启用全局反思和可学习分叉
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
        
        # 记录内存和时间
        import time
        import torch.cuda as cuda
        
        start_time = time.time()
        # 检测设备类型并重置内存统计
        if torch.backends.mps.is_available():
            device_type = "mps"
            # MPS不支持内存统计，使用0作为占位
            max_mem = 0
        elif cuda.is_available():
            device_type = "cuda"
            cuda.reset_peak_memory_stats()
        else:
            device_type = "cpu"
            max_mem = 0
        
        train_model(model, train_loader, val_loader, epochs=epochs)
        
        # 收集结果
        elapsed = time.time() - start_time
        if device_type == "cuda":
            max_mem = cuda.max_memory_allocated() / (1024 ** 2)
        elif device_type == "mps":
            max_mem = 0  # MPS内存统计不可用
        else:
            max_mem = 0
        
        results.append({
            "model": model_type,
            "train_time": elapsed,
            "max_memory": max_mem,
            "device": device_type
        })
    
    # 打印综合报告
    print("\n=== Benchmark Summary ===")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {epochs}")
    print("\nModel\tDevice\tTrain Time(s)\tMax Memory(MB)")
    for r in results:
        print(f"{r['model']}\t{r['device']}\t{r['train_time']:.2f}\t\t{r['max_memory']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="listops-32", 
                        choices=["listops-32", "wikitext-2"])
    parser.add_argument("--model", type=str, nargs="+", default=["pointer", "mamba", "performer", "transformer"],
                        choices=["pointer", "mamba", "performer", "transformer"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    # 如果传入单个模型，转换为列表
    if isinstance(args.model, str):
        args.model = [args.model]
    
    run_benchmark(args.dataset, args.model, args.epochs)
