import torch
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
import gc

# 混合精度训练支持
from torch.cuda.amp import autocast, GradScaler

def train_rl_matcher_fast(data_dir, num_epochs=50, batch_size=2, save_dir='./checkpoints', reward_type='distance', num_workers=4, grad_accum_steps=8, use_amp=True, memory_monitor=True, early_stop=True):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集，增加num_workers提升数据加载速度
    street_dataset = ImageFolder(os.path.join(data_dir, 'street'), transform)
    satellite_dataset = ImageFolder(os.path.join(data_dir, 'satellite'), transform)
    
    street_loader = DataLoader(street_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    satellite_loader = DataLoader(satellite_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    # 初始化RL模型
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用 Apple MPS 加速")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用 CUDA 加速")
        # 初始显存监控
        if memory_monitor and device.type == 'cuda':
            print(f"初始GPU显存占用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    else:
        device = torch.device('cpu')
        print("使用 CPU 运行")
    matcher = RLImageMatcher(feature_dim=2048, action_dim=len(satellite_dataset.classes))
    matcher.policy_net.to(device)
    matcher.target_net.to(device)
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    epochs = []
    rewards_history = []
    accuracy_history = []
    epsilon = 0.9
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
        total_reward = 0
        correct = 0
        total = 0
        losses = []
        epoch_loss = 0.0
        batch_loss_count = 0
        grad_accum_counter = 0
        
        # 清理显存和垃圾回收
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f'\n开始 Epoch {epoch+1}/{num_epochs}')
        print('=' * 50)
        
        # 显示当前显存使用情况
        if memory_monitor and device.type == 'cuda':
            print(f"Epoch {epoch+1} 开始时显存占用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        total_batches = min(len(street_loader), len(satellite_loader))
        pbar = tqdm(zip(street_loader, satellite_loader), total=total_batches, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100)
        
        for street_batch, satellite_batch in pbar:
            street_images, street_labels = street_batch
            satellite_images, satellite_labels = satellite_batch
            street_images = street_images.to(device, non_blocking=True)
            satellite_images = satellite_images.to(device, non_blocking=True)
            batch_reward = 0
            batch_correct = 0
            batch_total = 0
            batch_loss = 0.0
            batch_loss_items = 0
            
            for i in range(min(len(street_images), len(satellite_labels))):
                current_state = street_images[i].unsqueeze(0)
                # 混合精度训练
                if scaler is not None:
                    with autocast():
                        action = matcher.select_action(current_state)
                else:
                    action = matcher.select_action(current_state)
                action_idx = min(action.item(), len(satellite_images) - 1)
                reward = matcher.compute_reward(action_idx, satellite_labels[i].item(), reward_type)
                total_reward += reward
                batch_reward += reward
                next_state = satellite_images[action_idx].unsqueeze(0)
                matcher.memory.push(current_state, action, reward, next_state)
                if len(matcher.memory) >= batch_size:
                    if scaler is not None:
                        with autocast():
                            loss = matcher.update_model(batch_size=batch_size, scaler=scaler, grad_accum_steps=grad_accum_steps)
                        grad_accum_counter += 1
                        if grad_accum_counter % grad_accum_steps == 0:
                            # 梯度裁剪
                            scaler.unscale_(matcher.optimizer)
                            torch.nn.utils.clip_grad_norm_(matcher.policy_net.parameters(), 10)
                            scaler.step(matcher.optimizer)
                            scaler.update()
                            matcher.optimizer.zero_grad()
                    else:
                        loss = matcher.update_model(batch_size=batch_size, grad_accum_steps=grad_accum_steps)
                        grad_accum_counter += 1
                        if grad_accum_counter % grad_accum_steps == 0:
                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(matcher.policy_net.parameters(), 10)
                            matcher.optimizer.step()
                            matcher.optimizer.zero_grad()
                    losses.append(loss)
                    batch_loss += loss
                    batch_loss_items += 1
                    epoch_loss += loss
                    batch_loss_count += 1
                if action_idx == satellite_labels[i].item():
                    correct += 1
                    batch_correct += 1
                total += 1
                batch_total += 1
                
                # 每处理5个样本清理一次显存，更频繁地清理
                if i > 0 and i % 5 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
            # 检查是否需要早停
            if early_stop and device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated(device) / torch.cuda.max_memory_allocated(device) if torch.cuda.max_memory_allocated(device) > 0 else 0
                if current_memory > 0.95:
                    print(f"\n警告：显存使用率达到{current_memory:.2%}，触发早停机制")
                    print(f"在Epoch {epoch+1}提前停止训练并保存模型")
                    save_path = os.path.join(save_dir, f'rl_matcher_fast_epoch_{epoch+1}_early_stop.pth')
                    save_model(matcher.policy_net, save_path)
                    return matcher
            
            # 更新目标网络
            matcher.update_target_network()
            if batch_total > 0 and batch_loss_items > 0:
                pbar.set_postfix({
                    'loss': f'{batch_loss/batch_loss_items:.4f}',
                    'reward': f'{batch_reward/batch_total:.4f}',
                    'acc': f'{batch_correct/batch_total:.4f}',
                    'eps': f'{epsilon:.2f}'
                })
        matcher.update_scheduler()
        epsilon = max(0.1, epsilon * 0.95)
        accuracy = correct / total if total > 0 else 0
        avg_reward = total_reward / total if total > 0 else 0
        avg_loss = epoch_loss / batch_loss_count if batch_loss_count > 0 else 0
        print(f'\nEpoch {epoch+1}/{num_epochs} 完成 - 进度: {100*(epoch+1)/num_epochs:.1f}%')
        print(f'Average Reward: {avg_reward:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Epsilon: {epsilon:.4f}')
        print('-' * 50)
        epochs.append(epoch + 1)
        rewards_history.append(avg_reward)
        accuracy_history.append(accuracy)
        if (epoch + 1) % 5 == 0:
            # 保存模型前清理显存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # 显示保存前显存使用情况
            if memory_monitor and device.type == 'cuda':
                print(f"保存模型前显存占用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            
            # 将模型移至CPU再保存，避免GPU显存占用
            policy_net_cpu = matcher.policy_net.to('cpu')
            save_path = os.path.join(save_dir, f'rl_matcher_fast_epoch_{epoch+1}.pth')
            save_model(policy_net_cpu, save_path)
            # 保存后将模型移回原设备
            matcher.policy_net = policy_net_cpu.to(device)
            
            # 保存后再次清理
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # 显示保存后显存使用情况
            if memory_monitor and device.type == 'cuda':
                print(f"保存模型后显存占用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    # 清理显存和垃圾回收
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # 绘制训练曲线
    plot_training_curves(epochs, rewards_history, accuracy_history, save_path=os.path.join(save_dir, 'training_curves_fast.png'))
    
    # 最终显存使用情况
    if memory_monitor and device.type == 'cuda':
        print(f"训练结束后显存占用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
    return matcher

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='加速版强化学习图像匹配模型训练')
    parser.add_argument('--data_dir', type=str, default='./data/train', help='训练数据目录')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--reward_type', type=str, default='distance', choices=['binary', 'distance'], help='奖励函数类型')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--grad_accum_steps', type=int, default=8, help='梯度累积步数')
    parser.add_argument('--use_amp', action='store_true', help='是否使用混合精度')
    parser.add_argument('--memory_monitor', action='store_true', help='是否监控显存使用情况')
    parser.add_argument('--early_stop', action='store_true', help='是否启用显存监控早停机制')
    args = parser.parse_args()
    train_rl_matcher_fast(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        reward_type=args.reward_type,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        memory_monitor=args.memory_monitor,
        early_stop=args.early_stop
    )