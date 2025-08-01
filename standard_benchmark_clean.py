#!/usr/bin/env python3
"""
标准基准测试：ListOps-32 & WikiText-2
使用业界标准数据集进行严谨的模型对比

对比纯关系建模 vs 传统架构：
- ListOps-32: 结构化推理任务  
- WikiText-2: 语言建模任务
"""

import torch
import time
import numpy as np
import argparse
import sys
import os
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

# Add project paths
sys.path.append('/Volumes/oz/pointer')
sys.path.append('/Volumes/oz/pointer/benchmarks')

from standard_datasets import create_listops_dataloaders, create_wikitext_dataloaders
from fair_config import get_fair_training_config, print_fair_config, FAIR_HYPERPARAMS

# Import model trainers
from simple_trainers import (
    PointerTaskTrainer, TransformerTaskTrainer, MambaTaskTrainer, 
    PerformerTaskTrainer, TrainingConfig
)


def create_standard_config(task: str, model_size: str = "small", max_steps: int = None, enable_reflection: bool = False):
    """创建标准基准配置"""
    
    # 🎯 针对标准任务优化的配置
    if model_size == "tiny":
        base_config = {
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 8,
            'max_seq_len': 512,
            'batch_size': 16,
            'max_steps': max_steps or 3000,
            'eval_interval': 300,
        }
    elif model_size == "small":
        base_config = {
            'd_model': 384,
            'n_layers': 6, 
            'n_heads': 12,
            'max_seq_len': 512,
            'batch_size': 12,
            'max_steps': max_steps or 5000,
            'eval_interval': 200,
        }
    elif model_size == "medium":
        base_config = {
            'd_model': 512,
            'n_layers': 8,
            'n_heads': 16,
            'max_seq_len': 512,
            'batch_size': 8,
            'max_steps': max_steps or 8000,
            'eval_interval': 800,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # 根据任务调整配置
    if task == "listops":
        # ListOps需要更强的结构化推理能力
        base_config['top_k'] = 1  # 纯关系建模
        base_config['n_kv_heads'] = base_config['n_heads']  # 充分的关系建模能力
        vocab_size = 50  # ListOps词汇表较小
    elif task == "wikitext":
        # WikiText需要更强的语言建模能力
        base_config['top_k'] = 1  # 纯关系建模
        base_config['n_kv_heads'] = base_config['n_heads']
        vocab_size = 70000  # WikiText词汇表较大
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 系统配置
    device = 'cpu'  # Use CPU to avoid MPS compatibility issues
    fp16 = torch.cuda.is_available()
    
    config = TrainingConfig(
        vocab_size=vocab_size,
        **base_config,
        device=device,
        fp16=fp16,
        grad_checkpoint=False,
        output_dir=f'./standard_benchmark_{task}',
        experiment_name=f'standard_{task}_benchmark'
    )
    
    # 应用公平配置
    config = get_fair_training_config(config)
    
    # 反思机制配置
    if enable_reflection:
        reflection_layers = list(range(max(1, base_config['n_layers'] // 2), base_config['n_layers']))
        config.reflection_config = {
            'reflection_layers': reflection_layers,
            'pointer_backtrack_layers': 3,
            'reflection_gate_init': 0.1
        }
    else:
        config.reflection_config = {
            'reflection_layers': [],
            'pointer_backtrack_layers': 0,
            'reflection_gate_init': 0.0
        }
    
    return config


class StandardTaskTrainer:
    """标准任务的通用训练器"""
    
    def __init__(self, model_trainer, task_type: str):
        self.trainer = model_trainer
        self.task_type = task_type
        
    def train(self, train_loader, val_loader):
        """训练模型"""
        print(f"开始训练 {self.task_type} 任务...")
        
        self.trainer.train_step = 0
        best_metric = float('inf') if self.task_type == 'language_modeling' else 0.0
        
        for epoch in range(100):  # 最大epoch数
            epoch_loss = 0.0
            num_batches = 0
            
            self.trainer.model.train()
            for batch in train_loader:
                if self.trainer.train_step >= self.trainer.config.max_steps:
                    break
                
                # 准备输入数据
                input_ids = batch['input_ids'].to(self.trainer.device)
                labels = batch['labels'].to(self.trainer.device)
                attention_mask = batch.get('attention_mask')
                
                # 前向传播
                if hasattr(self.trainer.model, 'forward'):
                    outputs = self.trainer.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                else:
                    # 兼容不同的模型接口
                    loss = self.trainer.model(input_ids, labels=labels)
                
                # 反向传播
                self.trainer.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainer.model.parameters(), self.trainer.config.grad_clip_norm)
                self.trainer.optimizer.step()
                self.trainer.scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.trainer.train_step += 1
                
                # 定期评估
                if self.trainer.train_step % self.trainer.config.eval_interval == 0:
                    val_metrics = self.evaluate(val_loader)
                    
                    print(f"Step {self.trainer.train_step}:")
                    print(f"  Train Loss: {loss.item():.4f}")
                    
                    if self.task_type == 'classification':
                        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                        print(f"  Val Loss: {val_metrics['loss']:.4f}")
                        
                        # 保存最佳模型
                        if val_metrics['accuracy'] > best_metric:
                            best_metric = val_metrics['accuracy']
                            self.trainer.save_checkpoint('best_model.pt')
                    else:  # language_modeling
                        perplexity = np.exp(val_metrics['loss'])
                        print(f"  Val Loss: {val_metrics['loss']:.4f}")
                        print(f"  Val Perplexity: {perplexity:.2f}")
                        
                        # 保存最佳模型 (更低的loss更好)
                        if val_metrics['loss'] < best_metric:
                            best_metric = val_metrics['loss']
                            self.trainer.save_checkpoint('best_model.pt')
            
            if self.trainer.train_step >= self.trainer.config.max_steps:
                break
        
        print(f"训练完成！最佳指标: {best_metric:.4f}")
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.trainer.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.trainer.device)
                labels = batch['labels'].to(self.trainer.device)
                
                # 前向传播
                if hasattr(self.trainer.model, 'forward'):
                    outputs = self.trainer.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                    logits = outputs['logits']
                else:
                    # 兼容处理
                    result = self.trainer.model(input_ids, labels=labels)
                    loss = result if isinstance(result, torch.Tensor) else result.get('loss', result)
                    logits = getattr(result, 'logits', None)
                
                total_loss += loss.item()
                
                # 计算准确率 (仅分类任务)
                if self.task_type == 'classification' and logits is not None:
                    preds = torch.argmax(logits, dim=-1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        metrics = {'loss': avg_loss}
        
        if self.task_type == 'classification' and total_samples > 0:
            metrics['accuracy'] = total_correct / total_samples
        
        return metrics


def train_model_on_standard_task(model_type: str, task: str, config: TrainingConfig):
    """在标准任务上训练指定模型"""
    print(f"\n🚀 Training {model_type.upper()} on {task.upper()}...")
    
    # 创建数据加载器
    if task == "listops":
        data_loaders = create_listops_dataloaders('/Volumes/oz/pointer/listops-32', 
                                                 batch_size=config.batch_size, 
                                                 max_seq_len=config.max_seq_len)
        task_type = 'classification'
    elif task == "wikitext":
        data_loaders = create_wikitext_dataloaders('/Volumes/oz/pointer/wikitext-2',
                                                  batch_size=config.batch_size,
                                                  max_seq_len=config.max_seq_len)
        task_type = 'language_modeling'
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 更新配置中的词汇表大小
    config.vocab_size = data_loaders['vocab_size']
    
    # 创建对应的训练器
    if model_type == "pointer":
        trainer = PointerTaskTrainer(config, f"{task}_{model_type}")
    elif model_type == "transformer":
        trainer = TransformerTaskTrainer(config, f"{task}_{model_type}")
    elif model_type == "mamba":
        trainer = MambaTaskTrainer(config, f"{task}_{model_type}")
    elif model_type == "performer":
        trainer = PerformerTaskTrainer(config, f"{task}_{model_type}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 包装为标准任务训练器
    standard_trainer = StandardTaskTrainer(trainer, task_type)
    
    # 训练模型
    start_time = time.time()
    try:
        standard_trainer.train(data_loaders['train'], data_loaders['validation'])
        train_time = time.time() - start_time
        success = True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        train_time = time.time() - start_time
        success = False
        return None
    
    if not success:
        return None
    
    print(f"✅ {model_type} on {task} completed in {train_time:.1f}s")
    
    # 评估测试集
    test_metrics = standard_trainer.evaluate(data_loaders['test'])
    
    # 🎯 关系质量分析 (仅对Pointer模型)
    relation_metrics = {}
    if model_type == "pointer" and hasattr(trainer.model, 'get_pointer_stats'):
        relation_stats = trainer.model.get_pointer_stats()
        relation_metrics = {
            'relation_utilization': relation_stats.get('pointer_utilization', 0),
            'avg_hop_distance': relation_stats.get('avg_hop_distance', 0),
            'relation_entropy': relation_stats.get('pointer_entropy', 0)
        }
        
        print(f"🔗 关系质量指标:")
        for key, value in relation_metrics.items():
            print(f"   {key}: {value:.3f}")
    
    # 计算模型统计
    model_params = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    
    result = {
        'model_type': model_type,
        'task': task,
        'task_type': task_type,
        'train_time': train_time,
        'test_metrics': test_metrics,
        'relation_metrics': relation_metrics,
        'model_params': model_params,
        'vocab_size': data_loaders['vocab_size']
    }
    
    # 添加主要指标
    if task_type == 'classification':
        result['main_metric'] = test_metrics.get('accuracy', 0)
        result['metric_name'] = 'accuracy'
    else:
        result['main_metric'] = np.exp(test_metrics.get('loss', 10))  # perplexity
        result['metric_name'] = 'perplexity'
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Standard Dataset Benchmark (ListOps-32 & WikiText-2)')
    parser.add_argument('--models', nargs='+',
                       choices=['pointer', 'transformer', 'mamba', 'performer'],
                       default=['pointer', 'transformer'],
                       help='Models to compare')
    parser.add_argument('--model-size', choices=['tiny', 'small', 'medium'], default='small',
                       help='Model size configuration')
    parser.add_argument('--tasks', nargs='+', 
                       choices=['listops', 'wikitext'],
                       default=['listops'],
                       help='Standard tasks to benchmark')
    parser.add_argument('--max-steps', type=int, help='Override max training steps')
    
    args = parser.parse_args()
    
    print("🎯 STANDARD DATASET BENCHMARK")
    print("=" * 80)
    print("📊 Standard Tasks:", ", ".join(args.tasks))
    print("🤖 Models:", ", ".join(args.models))
    device_name = "MPS" if torch.backends.mps.is_available() else ("CUDA" if torch.cuda.is_available() else "CPU")
    print(f"💻 Device: {device_name}")
    print(f"📏 Model Size: {args.model_size}")
    print()
    
    # 显示公平配置
    print_fair_config()
    print()
    
    # 运行所有实验
    results = []
    total_experiments = len(args.tasks) * len(args.models)
    experiment_count = 0
    
    for task in args.tasks:
        for model in args.models:
            experiment_count += 1
            print(f"\n{'='*60}")
            print(f"🔬 Experiment {experiment_count}/{total_experiments}: {model.upper()} on {task.upper()}")
            print(f"{'='*60}")
            
            # 创建任务特定的配置
            config = create_standard_config(task, args.model_size, args.max_steps, False)
            
            try:
                result = train_model_on_standard_task(model, task, config)
                if result:
                    results.append(result)
                    metric_name = result['metric_name']
                    metric_value = result['main_metric']
                    print(f"✅ Success: {metric_name} = {metric_value:.4f}")
                else:
                    print("❌ Training failed")
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if not results:
        print("❌ No experiments completed successfully!")
        return
    
    print(f"\n🎯 Completed {len(results)}/{total_experiments} experiments successfully")
    
    # 打印详细结果
    print("\n" + "="*100)
    print("🎯 STANDARD BENCHMARK RESULTS")
    print("="*100)
    
    for result in results:
        model = result['model_type'].upper()
        task = result['task'].upper() 
        metric = result['main_metric']
        metric_name = result['metric_name']
        time_val = result['train_time']
        params = result['model_params']
        
        print(f"{model:<12} on {task:<8} | {metric_name}: {metric:.4f} | Time: {time_val:6.1f}s | Params: {params:.1f}M")
        
        # 显示关系质量指标（仅Pointer模型）
        if result['model_type'] == 'pointer' and result['relation_metrics']:
            rm = result['relation_metrics']
            print(f"             🔗 Relations: Util={rm['relation_utilization']:.3f} | Hop={rm['avg_hop_distance']:.2f} | Entropy={rm['relation_entropy']:.3f}")
    
    print("="*100)


if __name__ == "__main__":
    main()