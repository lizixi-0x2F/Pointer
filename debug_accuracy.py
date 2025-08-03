#!/usr/bin/env python3
"""
快速测试来诊断高准确率问题
"""
import torch
import sys
import os
sys.path.append('/Volumes/oz/Pointer')

from benchmarks.pointer_benchmark import load_dataset

def analyze_dataset_difficulty():
    """分析数据集难度来理解高准确率的原因"""
    print("=== WikiText-2 数据集难度分析 ===")
    
    # 加载数据集
    train, val, test, vocab_size, max_len, preprocess = load_dataset("wikitext-2")
    
    print(f"词汇表大小: {vocab_size}")
    print(f"最大序列长度: {max_len}")
    
    # 分析前几个样本
    sample_texts = []
    for i, item in enumerate(train):
        if i >= 5:
            break
        processed = preprocess(item, max_len)
        sample_texts.append(processed)
        print(f"\n样本 {i+1}:")
        print(f"原始文本: {item['text'][:100]}...")
        print(f"Token IDs: {processed['input_ids'][:20].tolist()}...")
        print(f"序列长度: {(processed['attention_mask']).sum().item()}")
    
    # 分析token分布
    print("\n=== Token频率分析 ===")
    all_tokens = []
    for i, item in enumerate(train):
        if i >= 1000:  # 只分析前1000个样本
            break
        processed = preprocess(item, max_len)
        tokens = processed['input_ids'].tolist()
        # 只计算非padding tokens
        mask = processed['attention_mask'].tolist()
        valid_tokens = [t for t, m in zip(tokens, mask) if m == 1]
        all_tokens.extend(valid_tokens)
    
    # 计算token频率
    from collections import Counter
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    print(f"总token数: {total_tokens}")
    print(f"唯一token数: {len(token_counts)}")
    print(f"词汇表利用率: {len(token_counts)/vocab_size:.2%}")
    
    # 最频繁的tokens
    print("\n最频繁的10个tokens:")
    for token, count in token_counts.most_common(10):
        char = chr(token) if 0 <= token <= 127 else f"[{token}]"
        print(f"  Token {token} ('{char}'): {count} 次 ({count/total_tokens:.2%})")
    
    # 计算理论随机准确率
    top1_freq = token_counts.most_common(1)[0][1] / total_tokens
    top5_freq = sum(count for _, count in token_counts.most_common(5)) / total_tokens
    
    print(f"\n=== 理论预测准确率 ===")
    print(f"随机猜测准确率: {1/vocab_size:.4f} ({100/vocab_size:.2f}%)")
    print(f"总是预测最频繁token: {top1_freq:.4f} ({top1_freq*100:.2f}%)")
    print(f"从top-5中随机选择: {top5_freq/5:.4f} ({top5_freq*20:.2f}%)")
    
    return token_counts, vocab_size

def test_next_char_predictability():
    """测试字符级别的下一个字符预测难度"""
    print("\n=== 字符级预测难度测试 ===")
    
    # 一些示例英文文本
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
    ]
    
    for text in sample_texts:
        print(f"\n文本: {text}")
        
        # 转换为字符tokens
        chars = [ord(c) for c in text]
        print(f"字符tokens: {chars[:10]}...")
        
        # 计算相邻字符的重复性
        bigram_count = 0
        total_pairs = len(chars) - 1
        for i in range(total_pairs):
            # 检查是否有常见的字符组合
            curr_char = chr(chars[i]) if chars[i] < 128 else "?"
            next_char = chr(chars[i+1]) if chars[i+1] < 128 else "?"
            if curr_char in "th" and next_char in "eaou":  # 'th' 后常跟元音
                bigram_count += 1
            elif curr_char == ' ' and next_char.isalpha():  # 空格后跟字母
                bigram_count += 1
            elif curr_char.isalpha() and next_char == ' ':  # 字母后跟空格
                bigram_count += 1
        
        predictability = bigram_count / total_pairs if total_pairs > 0 else 0
        print(f"本地模式可预测性: {predictability:.2%}")

if __name__ == "__main__":
    token_counts, vocab_size = analyze_dataset_difficulty()
    test_next_char_predictability()
    
    print(f"\n=== 结论 ===")
    print(f"1. 字符级词汇表很小 ({vocab_size} tokens)")
    print(f"2. 英文字符有很强的局部依赖性")
    print(f"3. 90%+的准确率在字符级任务中是可能的")
    print(f"4. 但验证集准确率高于训练集仍然暗示过拟合")