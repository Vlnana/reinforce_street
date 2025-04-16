import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rl_model import RLImageMatcher, DQN
from utils_rl import load_data, calculate_metrics, plot_confusion_matrix, load_model

def parse_args():
    parser = argparse.ArgumentParser(description='测试强化学习图像匹配模型')
    parser.add_argument('--data_dir', type=str, default='./data/test', help='测试数据目录')
    parser.add_argument('--model_path', type=str, default='rl_matcher_epoch_50.pth', help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--feature_dim', type=int, default=2048, help='特征维度')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--save_results', action='store_true', help='是否保存结果')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果输出目录')
    return parser.parse_args()

def test_rl_matcher(args):
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
    print(f"使用设备: {device}")
    
    # 加载数据
    street_loader, satellite_loader, num_classes = load_data(
        args.data_dir, batch_size=args.batch_size, shuffle=False
    )
    print(f"加载测试数据完成，类别数: {num_classes}")
    
    # 初始化模型
    model = DQN(feature_dim=args.feature_dim, action_dim=num_classes).to(device)
    model = load_model(model, args.model_path, device)
    
    # 创建RL匹配器（仅用于推理）
    matcher = RLImageMatcher(feature_dim=args.feature_dim, action_dim=num_classes)
    matcher.policy_net = model
    matcher.policy_net.eval()
    
    # 测试循环
    predictions = []
    ground_truth = []
    total_reward = 0
    total = 0
    
    print("开始测试...")
    with torch.no_grad():
        for street_batch, satellite_batch in zip(street_loader, satellite_loader):
            street_images, street_labels = street_batch
            satellite_images, satellite_labels = satellite_batch
            
            street_images = street_images.to(device)
            satellite_images = satellite_images.to(device)
            
            # 对每个街景图像进行匹配
            for i in range(len(street_images)):
                # 当前状态是街景图像
                current_state = street_images[i].unsqueeze(0)
                
                # 选择动作（预测匹配的卫星图像索引）
                action = matcher.select_action(current_state, epsilon=0.0)  # 测试时不使用探索
                
                # 计算奖励
                reward = matcher.compute_reward(action.item(), satellite_labels[i].item())
                total_reward += reward
                
                # 记录预测和真实标签
                predictions.append(action.item())
                ground_truth.append(satellite_labels[i].item())
                total += 1
    
    # 计算评估指标
    metrics = calculate_metrics(predictions, ground_truth)
    avg_reward = total_reward / total
    
    # 输出结果
    print("\n测试结果:")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    
    # 可视化结果
    if args.visualize or args.save_results:
        # 创建输出目录
        if args.save_results and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # 绘制混淆矩阵
        save_path = os.path.join(args.output_dir, 'confusion_matrix.png') if args.save_results else None
        plot_confusion_matrix(predictions, ground_truth, save_path=save_path)
        
        # 可以添加更多可视化，如t-SNE降维可视化特征分布等
    
    return metrics, avg_reward

def main():
    args = parse_args()
    metrics, avg_reward = test_rl_matcher(args)
    
    # 如果需要，可以将结果保存到文件
    if args.save_results:
        results = {
            'average_reward': avg_reward,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
        
        # 保存结果到文本文件
        with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

if __name__ == '__main__':
    main()