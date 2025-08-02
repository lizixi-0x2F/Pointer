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
from benchmarks.baselines import TransformerBaseline  # 只导入transformer
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
        max_len = 512  # 减少序列长度提高性能
        
        def preprocess(batch, max_len=512):
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
        
        # 简单预处理 - 使用字符级tokenization
        vocab_size = 256  # 使用ASCII字符集
        max_len = 512  # 减少序列长度提高性能
        
        def preprocess(batch, max_len=512):
            # 检查并适配不同字段名
            if "text" in batch:
                text = batch["text"]
            elif "input_ids" in batch:
                text = batch["input_ids"] 
            else:
                raise ValueError(f"Invalid batch format. Expected 'text' or 'input_ids', got: {batch.keys()}")
            
            # 处理文本字符串 - 转换为字符级token
            if isinstance(text, str):
                # 字符级tokenization: 每个字符转换为ASCII码
                tokens = [min(ord(c), 255) for c in text[:max_len]]  # 限制在0-255范围
            elif isinstance(text, (list, tuple)):
                tokens = text[:max_len]
            else:
                raise ValueError(f"Invalid text format. Expected str or list, got: {type(text)}")
            
            # Padding处理
            original_len = len(tokens)
            pad_len = max_len - len(tokens)
            if pad_len > 0:
                tokens = tokens + [0] * pad_len  # 用0填充
            
            # 确保所有tensor都是固定长度
            tokens = tokens[:max_len]  # 截断到max_len
            
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(tokens, dtype=torch.long),  # 语言建模任务：输入=标签
                "attention_mask": torch.tensor([1]*min(original_len, max_len) + [0]*max(0, max_len-original_len), dtype=torch.long)
            }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train, val, test, vocab_size, max_len, preprocess

def train_model(model, train_loader, val_loader, epochs=1, lr=0.003, return_metrics=False, max_steps=None):
    """训练模型并评估 - 支持按步数或epoch数训练"""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    model = model.to(device)
    
    # 跟踪训练指标
    training_metrics = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'pointer_stats': []  # 新增：指针统计信息
    }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    if max_steps is not None:
        print(f"Training for {max_steps} steps (1 epoch)")
        epochs = 1  # 强制使用1个epoch
    else:
        print(f"Training for {epochs} epochs")
    
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        if max_steps is not None:
            # 使用步数限制的进度条
            progress_bar = tqdm(total=max_steps, desc=f"Training Steps")
            data_iter = iter(train_loader)
            
            for step in range(max_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    print(f"\nDataLoader exhausted at step {global_step}, stopping training...")
                    break
                    
                # 统一的训练逻辑
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
                
                # 诊断梯度信息（每50步打印一次）
                if global_step % 50 == 0:
                    total_grad_norm = 0
                    param_count = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            total_grad_norm += grad_norm
                            param_count += 1
                    avg_grad_norm = total_grad_norm / max(param_count, 1)
                    print(f"  Step {global_step}: avg_grad_norm={avg_grad_norm:.6f}, lr={optimizer.param_groups[0]['lr']:.6f}")
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                global_step += 1
                
                # 更新进度条显示
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'Step': f"{global_step}/{max_steps}",
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*correct/total:.2f}%"
                })
                
        else:
            # 使用标准的epoch进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
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
                
                global_step += 1
                
                # 更新进度条显示
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*correct/total:.2f}%"
                })
        
        # 关闭进度条
        progress_bar.close()
        
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
        
        # 收集指针统计信息（如果是pointer模型）
        if hasattr(model, 'get_pointer_stats'):
            pointer_stats = model.get_pointer_stats()
            training_metrics['pointer_stats'].append(pointer_stats)
            if epoch == epochs - 1:  # 最后一个epoch打印详细统计
                print(f"Pointer Stats - Utilization: {pointer_stats.get('pointer_utilization', 0):.3f}, "
                      f"Avg Hop Distance: {pointer_stats.get('avg_hop_distance', 0):.2f}, "
                      f"Entropy: {pointer_stats.get('pointer_entropy', 0):.3f}")
        
        # 记录训练指标
        if global_step > 0:  # 确保有训练步数
            epoch_train_loss = train_loss/global_step if max_steps else train_loss/len(train_loader)
            epoch_train_acc = 100.*correct/total
            epoch_val_loss = val_loss/len(val_loader)
            epoch_val_acc = 100.*val_correct/val_total
            
            training_metrics['train_losses'].append(epoch_train_loss)
            training_metrics['train_accs'].append(epoch_train_acc)
            training_metrics['val_losses'].append(epoch_val_loss)
            training_metrics['val_accs'].append(epoch_val_acc)

            if max_steps is not None:
                print(f"Completed {global_step} steps: "
                      f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% | "
                      f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
            else:
                print(f"Epoch {epoch+1}: "
                      f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% | "
                      f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # 如果达到最大步数，提前结束
        if max_steps is not None and global_step >= max_steps:
            break
    
    if return_metrics:
        return training_metrics

def run_benchmark(dataset_name, models=["pointer", "transformer"], epochs=10, detailed_analysis=False, max_steps=1000):
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
        batch_size=32,  # 增加批量大小
        shuffle=True
    )
    val_loader = DataLoader(
        val_processed,
        batch_size=32  # 增加批量大小
    )
    test_loader = DataLoader(
        test_processed,
        batch_size=32  # 增加批量大小
    )
    
    results = []
    
    for model_type in models:
        # 初始化模型
        if model_type == "pointer":
            # 增强的双向多头指针配置
            reflection_config = {
                # 全局反思机制，无需指定特定层
                'global_reflection': True,
                'bidirectional_multihead': True,  # 启用双向多头指针
                'learnable_parameters': True      # 确保所有参数可学习
            }
            
            model = PointerDecoder(
                vocab_size=vocab_size,
                d=128,
                n_layers=4,
                n_heads=2,  # 减少头数从4到2
                max_seq_len=max_len,
                reflection_config=reflection_config  # 启用全局反思和双向多头指针
            )
        elif model_type == "transformer":
            model = TransformerBaseline(vocab_size=vocab_size, d_model=128, n_layers=4)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: ['pointer', 'transformer']")
        
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
        
        train_metrics = train_model(model, train_loader, val_loader, epochs=epochs, return_metrics=detailed_analysis, max_steps=max_steps)
        
        # 收集结果
        elapsed = time.time() - start_time
        if device_type == "cuda":
            max_mem = cuda.max_memory_allocated() / (1024 ** 2)
        elif device_type == "mps":
            max_mem = 0  # MPS内存统计不可用
        else:
            max_mem = 0
        
        # 收集详细结果
        model_result = {
            "model": model_type,
            "train_time": elapsed,
            "max_memory": max_mem,
            "device": device_type
        }
        
        # 如果有训练指标，添加最终性能
        if detailed_analysis and train_metrics and len(train_metrics.get('train_accs', [])) > 0:
            model_result.update({
                "final_train_acc": train_metrics['train_accs'][-1],
                "final_val_acc": train_metrics['val_accs'][-1],
                "final_train_loss": train_metrics['train_losses'][-1],
                "final_val_loss": train_metrics['val_losses'][-1]
            })
            
            # 指针模型的特殊统计
            if model_type == "pointer" and train_metrics.get('pointer_stats') and len(train_metrics['pointer_stats']) > 0:
                final_pointer_stats = train_metrics['pointer_stats'][-1]
                model_result.update({
                    "pointer_utilization": final_pointer_stats.get('pointer_utilization', 0),
                    "avg_hop_distance": final_pointer_stats.get('avg_hop_distance', 0),
                    "pointer_entropy": final_pointer_stats.get('pointer_entropy', 0)
                })
        else:
            # 添加默认值防止错误
            if detailed_analysis:
                model_result.update({
                    "final_train_acc": 0.0,
                    "final_val_acc": 0.0,
                    "final_train_loss": 0.0,
                    "final_val_loss": 0.0
                })
        
        results.append(model_result)
    
    # 打印增强的综合报告
    print("\n=== Enhanced Benchmark Summary ===")
    print(f"Dataset: {dataset_name}")
    if max_steps is not None:
        print(f"Training: {max_steps} steps (1 epoch)")
    else:
        print(f"Training: {epochs} epochs")
    print(f"Architecture Updates: Bidirectional Multi-Head Pointers, Learnable Parameters")
    
    if detailed_analysis:
        print("\nDetailed Performance Analysis:")
        print("Model\t\tDevice\tTime(s)\tMemory(MB)\tTrain Acc\tVal Acc\tTrain Loss\tVal Loss")
        for r in results:
            print(f"{r['model']:<12}\t{r['device']}\t{r['train_time']:.1f}\t{r['max_memory']:.1f}\t\t"
                  f"{r.get('final_train_acc', 0):.2f}%\t\t{r.get('final_val_acc', 0):.2f}%\t\t"
                  f"{r.get('final_train_loss', 0):.4f}\t\t{r.get('final_val_loss', 0):.4f}")
        
        # 指针模型特殊分析
        pointer_results = [r for r in results if r['model'] == 'pointer']
        if pointer_results:
            print("\nPointer Architecture Analysis:")
            for r in pointer_results:
                if 'pointer_utilization' in r:
                    print(f"- Pointer Utilization: {r['pointer_utilization']:.3f} (non-self pointing ratio)")
                    print(f"- Average Hop Distance: {r['avg_hop_distance']:.2f} tokens")
                    print(f"- Pointer Entropy: {r['pointer_entropy']:.3f} (relationship diversity)")
    else:
        print("\nBasic Performance Summary:")
        print("Model\t\tDevice\tTrain Time(s)\tMax Memory(MB)")
        for r in results:
            print(f"{r['model']:<12}\t{r['device']}\t{r['train_time']:.2f}\t\t{r['max_memory']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="listops-32", 
                        choices=["listops-32", "wikitext-2"])
    parser.add_argument("--model", type=str, nargs="+", default=["pointer", "transformer"],
                        choices=["pointer", "transformer"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum training steps (default: 1000)")
    parser.add_argument("--detailed", action="store_true", 
                        help="Enable detailed analysis including pointer statistics")
    args = parser.parse_args()
    
    # 如果传入单个模型，转换为列表
    if isinstance(args.model, str):
        args.model = [args.model]
    
    run_benchmark(args.dataset, args.model, args.epochs, detailed_analysis=args.detailed, max_steps=args.max_steps)
