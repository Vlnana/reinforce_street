import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 数据预处理函数
def get_data_transforms():
    """
    返回用于图像预处理的transforms
    """
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 数据加载函数
def load_data(data_dir, batch_size=32, shuffle=True):
    """
    加载街景和卫星图像数据集
    
    参数:
        data_dir: 数据目录路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    返回:
        street_loader: 街景图像数据加载器
        satellite_loader: 卫星图像数据加载器
        num_classes: 类别数量
    """
    transform = get_data_transforms()
    
    street_dataset = ImageFolder(os.path.join(data_dir, 'street'), transform)
    satellite_dataset = ImageFolder(os.path.join(data_dir, 'satellite'), transform)
    
    street_loader = DataLoader(street_dataset, batch_size=batch_size, shuffle=shuffle)
    satellite_loader = DataLoader(satellite_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return street_loader, satellite_loader, len(satellite_dataset.classes)

# 奖励函数
def compute_reward(predicted_idx, true_idx, reward_type='distance'):
    """
    计算强化学习的奖励值
    
    参数:
        predicted_idx: 预测的索引
        true_idx: 真实的索引
        reward_type: 奖励类型，可选 'binary'(二元奖励) 或 'distance'(基于距离的奖励)
        
    返回:
        reward: 奖励值
    """
    if reward_type == 'binary':
        # 简单的二元奖励：正确匹配得到正奖励，错误得到负奖励
        return 1.0 if predicted_idx == true_idx else -0.1
    elif reward_type == 'distance':
        # 基于距离的奖励：正确匹配得到最高奖励，错误匹配根据距离给予不同程度的负奖励
        if predicted_idx == true_idx:
            return 1.0
        else:
            # 计算索引距离，假设索引值接近表示图像相似度更高
            # 归一化距离到[0,1]范围，假设最大可能距离是类别数量
            max_distance = 951  # 假设的最大类别数
            distance = abs(predicted_idx - true_idx)
            normalized_distance = min(distance / max_distance, 1.0)
            
            # 根据距离计算奖励：距离越近，负奖励越小
            # 映射到[-0.5, -0.01]范围
            reward = -0.5 * normalized_distance - 0.01
            return reward
    elif reward_type == 'similarity':
        # 基于相似度的奖励：使用余弦相似度或其他相似度度量
        # 这需要额外的特征向量作为输入，此处仅为示例
        if predicted_idx == true_idx:
            return 1.0
        else:
            # 简化版：根据索引差异的倒数计算相似度
            similarity = 1.0 / (1.0 + abs(predicted_idx - true_idx))
            # 映射到[-0.3, 0.5]范围
            reward = similarity - 0.3
            return reward
    else:
        raise ValueError(f"不支持的奖励类型: {reward_type}")

# 评估指标计算函数
def calculate_metrics(predictions, ground_truth):
    """
    计算模型性能指标
    
    参数:
        predictions: 预测结果列表
        ground_truth: 真实标签列表
        
    返回:
        metrics: 包含各种指标的字典
    """
    accuracy = sum(p == gt for p, gt in zip(predictions, ground_truth)) / len(predictions)
    
    # 计算精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='macro'
    )
    
    # 计算Top-K准确率（如果predictions是概率分布）
    # 这里简化为只返回Top-1准确率
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 可视化函数
def plot_training_curves(epochs, rewards, accuracies, save_path=None):
    """
    绘制训练过程中的奖励和准确率曲线
    
    参数:
        epochs: 轮次列表
        rewards: 奖励列表
        accuracies: 准确率列表
        save_path: 保存图像的路径，如果为None则显示图像
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, rewards, 'b-')
    plt.title('Average Reward per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'r-')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 混淆矩阵可视化
def plot_confusion_matrix(predictions, ground_truth, class_names=None, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        predictions: 预测结果列表
        ground_truth: 真实标签列表
        class_names: 类别名称列表，如果为None则使用索引
        save_path: 保存图像的路径，如果为None则显示图像
    """
    cm = confusion_matrix(ground_truth, predictions)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 保存和加载模型函数
def save_model(model, path):
    """
    保存模型到指定路径
    
    参数:
        model: 要保存的模型
        path: 保存路径
    """
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")

def load_model(model, path, device=None):
    """
    从指定路径加载模型
    
    参数:
        model: 要加载权重的模型
        path: 模型权重路径
        device: 设备（'mps'、'cuda'或'cpu'）
        
    返回:
        加载了权重的模型
    """
    if device is None:
        # 设置设备优先级：1. MPS (Apple Silicon)  2. CUDA  3. CPU
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("使用 Apple MPS 加速")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("使用 CUDA 加速")
        else:
            device = torch.device('cpu')
            print("使用 CPU 运行")
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已从 {path} 加载")
    return model