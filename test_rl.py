import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rl_model import RLImageMatcher, DQN
from utils_rl import load_data, calculate_metrics, plot_confusion_matrix, load_model
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='测试强化学习图像匹配模型')
    parser.add_argument('--data_dir', type=str, default='./data/masked_test_set', help='测试数据目录')
    parser.add_argument('--model_path', type=str, default='./checkpoints/rl_matcher_epoch_5.pth', help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--feature_dim', type=int, default=2048, help='特征维度')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--save_results', action='store_true', help='是否保存结果')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果输出目录')
    parser.add_argument('--query_name', type=str, default='query_street_name.txt', help='查询图像名称文件')
    return parser.parse_args()

def get_SatId_160k(img_path):
    """获取卫星图像ID和路径"""
    labels = []
    paths = []
    for path, v in img_path:
        labels.append(v)
        paths.append(path)
    return labels, paths

def get_result_rank10(qf, gf, gl):
    """获取前10个匹配结果"""
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    rank10_index = index[0:10]
    result_rank10 = gl[rank10_index]
    return result_rank10

def load_custom_data(data_dir, query_name, batch_size=32):
    """加载自定义数据集"""
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 使用与test_demo.py相同的数据加载方式
    from image_folder import CustomData160k_sat, CustomData160k_drone
    
    image_datasets = {}
    image_datasets['gallery_satellite'] = CustomData160k_sat(os.path.join(data_dir, 'workshop_gallery_satellite'), transform)
    image_datasets['query_street'] = CustomData160k_drone(os.path.join(data_dir, 'workshop_query_street'), transform, query_name=query_name)
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) 
                  for x in ['gallery_satellite', 'query_street']}
    
    return dataloaders, image_datasets

def extract_feature(model, dataloader, device):
    """提取特征"""
    features = torch.FloatTensor()
    count = 0
    
    # 正确迭代DataLoader对象
    for img, label in dataloader:
        n, c, h, w = img.size()
        count += n
        print(f"处理 {count} 张图像")
        
        # 将图像移动到设备上
        img = img.to(device)
        
        # 提取特征
        with torch.no_grad():
            outputs = model(img)
            
            # 标准化特征
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            outputs = outputs.div(fnorm.expand_as(outputs))
            
            features = torch.cat((features, outputs.cpu()), 0)
    
    return features

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
    
    # 加载自定义数据集
    dataloaders, image_datasets = load_custom_data(
        args.data_dir, args.query_name, batch_size=args.batch_size
    )
    
    # 获取卫星图像标签
    gallery_path = image_datasets['gallery_satellite'].imgs
    gallery_label, gallery_paths = get_SatId_160k(gallery_path)
    gallery_label = np.array(gallery_label)
    
    # 初始化模型
    num_classes = len(image_datasets['gallery_satellite'].classes)
    model = DQN(feature_dim=args.feature_dim, action_dim=num_classes).to(device)
    model = load_model(model, args.model_path, device)
    model.eval()
    
    print("开始提取特征...")
    with torch.no_grad():
        print('-------------------提取查询特征----------------------')
        query_feature = extract_feature(model, dataloaders['query_street'], device)
        print('-------------------提取图库特征----------------------')
        gallery_feature = extract_feature(model, dataloaders['gallery_satellite'], device)
        print('--------------------------特征提取完成-------------------------------')
    
    # 将特征移动到设备上进行计算
    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)
    
    # 生成排名前10的结果
    save_filename = 'results_rank10.txt'
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    
    results_rank10 = []
    print(f"处理 {len(query_feature)} 个查询")
    
    for i in range(len(query_feature)):
        result_rank10 = get_result_rank10(query_feature[i], gallery_feature, gallery_label)
        results_rank10.append(result_rank10)
    
    # 保存结果
    results_rank10 = np.row_stack(results_rank10)
    with open(save_filename, 'w') as f:
        for row in results_rank10:
            f.write('\t'.join(map(str, row)) + '\n')
    
    print(f"结果已保存到 {save_filename}")
    
    # 计算评估指标（如果有真实标签）
    metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    
    return metrics, 0

def main():
    args = parse_args()
    
    # 创建输出目录
    if args.save_results and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 运行测试
    metrics, avg_reward = test_rl_matcher(args)
    
    print("\n测试完成！")
    print(f"结果已保存到 results_rank10.txt")
    
    # 如果需要，可以将评估指标保存到文件
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