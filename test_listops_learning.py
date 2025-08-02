import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.model.pointer_model import PointerDecoder
from src.layers.embedding import TokenEmbedding
from tqdm import tqdm

def create_simple_listops_data(num_samples=1000, seq_len=10, vocab_size=20):
    """创建简单的ListOps风格数据"""
    # 生成随机序列
    X = torch.randint(1, vocab_size, (num_samples, seq_len))
    # 简单规则：序列中最大值的索引作为标签
    y = torch.argmax(X, dim=1)
    return X, y

def test_listops_learning():
    # 超参数
    d_model = 128
    n_layers = 4
    n_heads = 8
    top_k = 2
    batch_size = 16
    lr = 0.001
    epochs = 10
    
    vocab_size = 10  # 统一vocab_size为10
    # 创建数据
    X_train, y_train = create_simple_listops_data(1000, vocab_size=vocab_size)
    X_val, y_val = create_simple_listops_data(200, vocab_size=vocab_size)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # 初始化模型
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = PointerDecoder(
        vocab_size=10,  # 统一vocab_size为10
        d=128,
        n_layers=4,
        n_heads=4
    ).to(device)
    model.device = device  # 添加device属性
    
    # 训练设置
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            
            outputs = model(inputs)
            # 确保使用正确的logits格式
            logits = outputs['logits']
            if logits.dim() == 3:  # 如果是[B, 1, vocab_size]
                logits = logits.squeeze(1)  # 变为[B, vocab_size]
            elif logits.dim() == 2:  # 已经是[B, vocab_size]
                pass
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            labels = labels.view(-1).long()
            # 检查维度是否匹配
            assert logits.size(0) == labels.size(0), f"Batch size mismatch: logits {logits.size()}, labels {labels.size()}"
            assert logits.size(1) == 10, f"Vocab size mismatch: expected 10, got {logits.size(1)}"
            # 确保logits是[B, C]格式，labels是[B]格式
            if logits.dim() == 3:  # [B, N, C] -> [B, C]
                logits = logits.mean(dim=1)  # 取平均或选择特定位置
            elif logits.dim() != 2:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            if labels.dim() != 1:
                labels = labels.view(-1)
            
            # 检查维度是否匹配
            if logits.size(0) != labels.size(0):
                raise ValueError(f"Batch size mismatch: logits {logits.shape}, labels {labels.shape}")
            
            # 确保logits是[B, C]格式，labels是[B]格式
            if logits.dim() == 3:  # [B, 1, C] -> [B, C]
                logits = logits.squeeze(1)
            elif logits.dim() != 2:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            if labels.dim() != 1:
                labels = labels.view(-1)
            
            # 检查维度是否匹配
            if logits.size(0) != labels.size(0):
                raise ValueError(f"Batch size mismatch: logits {logits.shape}, labels {labels.shape}")
            
            # 确保logits是[B, C]格式，labels是[B]格式
            if logits.dim() == 3:  # [B, 1, C] -> [B, C]
                logits = logits.squeeze(1)
            elif logits.dim() != 2:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            if labels.dim() != 1:
                labels = labels.view(-1)
            
            # 检查维度是否匹配
            if logits.size(0) != labels.size(0):
                raise ValueError(f"Batch size mismatch: logits {logits.shape}, labels {labels.shape}")
            
            # 打印调试信息
            print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
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
            for inputs, labels in val_loader:
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                
                outputs = model(inputs)
                # 确保使用分类任务的logits输出
                logits = outputs['logits']
                # 确保logits是[B, C]格式，labels是[B]格式
                if logits.dim() == 3:  # [B, N, C] -> [B, C]
                    logits = logits.mean(dim=1)
                elif logits.dim() != 2:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                
                labels = labels.view(-1).long()
                # 检查维度是否匹配
                if logits.size(0) != labels.size(0):
                    raise ValueError(f"Batch size mismatch: logits {logits.shape}, labels {labels.shape}")
                
                print(f"Val Logits shape: {logits.shape}, Labels shape: {labels.shape}")
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}% | '
              f'Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {100.*val_correct/val_total:.2f}%')

if __name__ == '__main__':
    test_listops_learning()
