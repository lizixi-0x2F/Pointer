"""
Standard Dataset Loaders for ListOps-32 and WikiText-2
专门处理标准基准数据集的加载器
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import re


class ListOpsDataset(Dataset):
    """
    ListOps-32 数据集：结构化推理任务
    
    任务：解析嵌套的列表操作表达式，如：
    [MIN 1 [SM 3 6 4 0 5] 5 [MED 6 3 8] 0] -> 0
    
    这是测试关系建模和多步推理的理想任务。
    """
    
    def __init__(self, parquet_path: str, max_seq_len: int = 512):
        self.df = pd.read_parquet(parquet_path)
        self.max_seq_len = max_seq_len
        
        # 构建词汇表
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        print(f"ListOps loaded: {len(self.df)} examples, vocab_size: {self.vocab_size}")
        
    def _build_vocab(self):
        """构建词汇表"""
        vocab = set()
        
        for source in self.df['Source']:
            # 分词：空格分割，保留括号
            tokens = re.findall(r'\[|\]|\w+', source)
            vocab.update(tokens)
        
        # 添加特殊token
        vocab_list = ['<pad>', '<unk>', '<bos>', '<eos>'] + sorted(list(vocab))
        return {token: idx for idx, token in enumerate(vocab_list)}
    
    def _tokenize(self, text: str) -> List[int]:
        """分词并转换为token IDs"""
        tokens = re.findall(r'\[|\]|\w+', text)
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        # 添加BOS/EOS并截断
        token_ids = [self.vocab['<bos>']] + token_ids + [self.vocab['<eos>']]
        
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len-1] + [self.vocab['<eos>']]
        
        return token_ids
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source = row['Source']
        target = int(row['Target'])
        
        # 分词
        input_ids = self._tokenize(source)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(target, dtype=torch.long),
            'length': len(input_ids)
        }


class WikiTextDataset(Dataset):
    """
    WikiText-2 数据集：语言建模任务
    
    任务：预测下一个token
    这是测试长文本理解和生成的标准任务。
    """
    
    def __init__(self, parquet_path: str, max_seq_len: int = 512, stride: int = 256):
        self.df = pd.read_parquet(parquet_path)
        self.max_seq_len = max_seq_len
        self.stride = stride
        
        # 构建词汇表并准备序列
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.sequences = self._prepare_sequences()
        
        print(f"WikiText-2 loaded: {len(self.sequences)} sequences, vocab_size: {self.vocab_size}")
    
    def _build_vocab(self):
        """构建词汇表"""
        vocab = set()
        
        for text in self.df['text']:
            # 简单的词级分词
            tokens = text.lower().split()
            vocab.update(tokens)
        
        # 添加特殊token
        vocab_list = ['<pad>', '<unk>', '<bos>', '<eos>'] + sorted(list(vocab))
        return {token: idx for idx, token in enumerate(vocab_list)}
    
    def _tokenize(self, text: str) -> List[int]:
        """分词并转换为token IDs"""
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return token_ids
    
    def _prepare_sequences(self):
        """准备训练序列（滑动窗口）"""
        sequences = []
        
        for text in self.df['text']:
            if len(text.strip()) == 0:
                continue
                
            token_ids = self._tokenize(text)
            
            # 滑动窗口切分
            for i in range(0, len(token_ids), self.stride):
                seq = token_ids[i:i + self.max_seq_len]
                if len(seq) >= 32:  # 最小序列长度
                    sequences.append(seq)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 语言建模：输入是seq[:-1]，标签是seq[1:]
        input_ids = seq[:-1]  
        labels = seq[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'length': len(input_ids)
        }


def collate_fn(batch):
    """批处理函数：padding到相同长度"""
    max_len = max(item['length'] for item in batch)
    
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        input_length = item['length']
        pad_length = max_len - input_length
        
        # Pad input_ids
        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(pad_length, dtype=torch.long)
        ])
        input_ids.append(padded_input)
        
        # Handle labels (different for ListOps vs WikiText)
        if item['labels'].dim() == 0:  # ListOps: scalar label
            labels.append(item['labels'])
        else:  # WikiText: sequence labels
            padded_labels = torch.cat([
                item['labels'],
                torch.full((pad_length,), -100, dtype=torch.long)  # -100 ignored in loss
            ])
            labels.append(padded_labels)
        
        # Attention mask
        mask = torch.cat([
            torch.ones(input_length, dtype=torch.bool),
            torch.zeros(pad_length, dtype=torch.bool)
        ])
        attention_mask.append(mask)
    
    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask)
    }
    
    # Handle different label types
    if batch[0]['labels'].dim() == 0:  # ListOps
        result['labels'] = torch.stack(labels)
        result['task_type'] = 'classification'
    else:  # WikiText
        result['labels'] = torch.stack(labels)
        result['task_type'] = 'language_modeling'
    
    return result


def create_listops_dataloaders(data_dir: str, batch_size: int = 16, max_seq_len: int = 512):
    """创建ListOps数据加载器"""
    import os
    
    train_dataset = ListOpsDataset(
        os.path.join(data_dir, 'data/train-00000-of-00001.parquet'),
        max_seq_len=max_seq_len
    )
    
    val_dataset = ListOpsDataset(
        os.path.join(data_dir, 'data/validation-00000-of-00001.parquet'),
        max_seq_len=max_seq_len
    )
    
    test_dataset = ListOpsDataset(
        os.path.join(data_dir, 'data/test-00000-of-00001.parquet'),
        max_seq_len=max_seq_len
    )
    
    # 确保所有数据集使用相同的词汇表
    val_dataset.vocab = train_dataset.vocab
    val_dataset.vocab_size = train_dataset.vocab_size
    test_dataset.vocab = train_dataset.vocab
    test_dataset.vocab_size = train_dataset.vocab_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader,
        'vocab_size': train_dataset.vocab_size
    }


def create_wikitext_dataloaders(data_dir: str, batch_size: int = 16, max_seq_len: int = 512):
    """创建WikiText-2数据加载器"""
    import os
    
    train_dataset = WikiTextDataset(
        os.path.join(data_dir, 'data/train-00000-of-00001.parquet'),
        max_seq_len=max_seq_len
    )
    
    val_dataset = WikiTextDataset(
        os.path.join(data_dir, 'data/validation-00000-of-00001.parquet'),
        max_seq_len=max_seq_len
    )
    
    test_dataset = WikiTextDataset(
        os.path.join(data_dir, 'data/test-00000-of-00001.parquet'),
        max_seq_len=max_seq_len
    )
    
    # 确保所有数据集使用相同的词汇表
    val_dataset.vocab = train_dataset.vocab
    val_dataset.vocab_size = train_dataset.vocab_size
    test_dataset.vocab = train_dataset.vocab
    test_dataset.vocab_size = train_dataset.vocab_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader,
        'vocab_size': train_dataset.vocab_size
    }


if __name__ == "__main__":
    # 测试数据加载器
    print("Testing ListOps dataset...")
    listops_loaders = create_listops_dataloaders('/Volumes/oz/pointer/listops-32', batch_size=4)
    
    for batch in listops_loaders['train']:
        print(f"ListOps batch shape: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
        print(f"Task type: {batch['task_type']}")
        break
    
    print("\nTesting WikiText dataset...")
    wiki_loaders = create_wikitext_dataloaders('/Volumes/oz/pointer/wikitext-2', batch_size=4)
    
    for batch in wiki_loaders['train']:
        print(f"WikiText batch shape: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
        print(f"Task type: {batch['task_type']}")
        break
    
    print("✅ Standard datasets loaded successfully!")