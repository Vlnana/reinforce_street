#
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl_model import RLImageMatcher
from utils_rl import plot_training_curves, save_model

def train_rl_matcher(data_dir, num_epochs=50, batch_size=32, save_dir='./checkpoints', reward_type='distance'):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    street_dataset = ImageFolder(os.path.join(data_dir, 'street'), transform)
    satellite_dataset = ImageFolder(os.path.join(data_dir, 'satellite'), transform)
    
    street_loader = DataLoader(street_dataset, batch_size=batch_size, shuffle=True)
    satellite_loader = DataLoader(satellite_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化RL模型
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
    matcher = RLImageMatcher(feature_dim=2048, action_dim=len(satellite_dataset.classes))
    matcher.policy_net.to(device)
    matcher.target_net.to(device)
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 训练循环
    epochs = []
    rewards_history = []
    accuracy_history = []
    epsilon = 0.9  # 初始探索率
    
    for epoch in range(num_epochs):
        total_reward = 0
        correct = 0
        total = 0
        losses = []
        epoch_loss = 0.0
        batch_loss_count = 0
        
        # 显示当前epoch开始
        print(f'\n开始 Epoch {epoch+1}/{num_epochs}')
        print('=' * 50)
        
        # 跟踪batch进度
        total_batches = min(len(street_loader), len(satellite_loader))
        
        # 使用tqdm创建进度条
        pbar = tqdm(zip(street_loader, satellite_loader), total=total_batches, 
                   desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100)
        
        for street_batch, satellite_batch in pbar:
            street_images, street_labels = street_batch
            satellite_images, satellite_labels = satellite_batch
            
            street_images = street_images.to(device)
            satellite_images = satellite_images.to(device)
            
            batch_reward = 0
            batch_correct = 0
            batch_total = 0
            batch_loss = 0.0
            batch_loss_items = 0
            
            # 对每个街景图像进行匹配
            for i in range(min(len(street_images), len(satellite_labels))):
                # 当前状态是街景图像
                current_state = street_images[i].unsqueeze(0)
                
                # 选择动作（预测匹配的卫星图像索引）
                action = matcher.select_action(current_state)
                
                # 确保action不超出当前batch的范围
                action_idx = min(action.item(), len(satellite_images) - 1)
                
                # 计算奖励
                reward = matcher.compute_reward(action_idx, satellite_labels[i].item(), reward_type)
                total_reward += reward
                batch_reward += reward
                
                # 存储经验
                next_state = satellite_images[action_idx].unsqueeze(0)
                matcher.memory.push(current_state, action, reward, next_state)
                
                # 更新模型
                if len(matcher.memory) >= batch_size:
                    loss = matcher.update_model(batch_size=batch_size)
                    losses.append(loss)
                    batch_loss += loss
                    batch_loss_items += 1
                    epoch_loss += loss
                    batch_loss_count += 1
                
                # 统计准确率
                if action_idx == satellite_labels[i].item():
                    correct += 1
                    batch_correct += 1
                total += 1
                batch_total += 1
            
            # 每个批次都软更新目标网络
            matcher.update_target_network()
            
            # 更新进度条信息
            if batch_total > 0 and batch_loss_items > 0:
                pbar.set_postfix({
                    'loss': f'{batch_loss/batch_loss_items:.4f}',
                    'reward': f'{batch_reward/batch_total:.4f}',
                    'acc': f'{batch_correct/batch_total:.4f}',
                    'eps': f'{epsilon:.2f}'
                })
        
        # 更新学习率
        matcher.update_scheduler()
        
        # 减小探索率
        epsilon = max(0.1, epsilon * 0.95)  # 探索率衰减
        
        # 输出训练信息
        accuracy = correct / total if total > 0 else 0
        avg_reward = total_reward / total if total > 0 else 0
        avg_loss = epoch_loss / batch_loss_count if batch_loss_count > 0 else 0
        
        # 显示当前epoch进度
        print(f'\nEpoch {epoch+1}/{num_epochs} 完成 - 进度: {100*(epoch+1)/num_epochs:.1f}%')
        print(f'Average Reward: {avg_reward:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Epsilon: {epsilon:.4f}')
        print('-' * 50)
        
        # 记录历史数据用于绘图
        epochs.append(epoch + 1)
        rewards_history.append(avg_reward)
        accuracy_history.append(accuracy)
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_dir, f'rl_matcher_epoch_{epoch+1}.pth')
            save_model(matcher.policy_net, save_path)

    # 绘制训练曲线
    plot_training_curves(epochs, rewards_history, accuracy_history, 
                         save_path=os.path.join(save_dir, 'training_curves.png'))
    
    return matcher

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练强化学习图像匹配模型')
    parser.add_argument('--data_dir', type=str, default='./data/train', help='训练数据目录')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--reward_type', type=str, default='distance', choices=['binary', 'distance'], help='奖励函数类型')
    
    args = parser.parse_args()
    
    train_rl_matcher(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        reward_type=args.reward_type
    )