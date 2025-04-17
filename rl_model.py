import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import numpy as np
from utils_rl import compute_reward

class DQN(nn.Module):
    def __init__(self, feature_dim=2048, action_dim=951):
        super(DQN, self).__init__()
        # 使用ResNet50作为特征提取器，冻结更多层以减少显存使用
        resnet = models.resnet50(pretrained=True)
        # 冻结更多层参数，只训练最后几层
        for param in list(resnet.parameters())[:-10]:  # 从-20改为-10，冻结更多层
            param.requires_grad = False
            
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # 减小Q网络规模，降低中间层神经元数量以减少显存占用
        self.q_net = nn.Sequential(
            nn.Linear(feature_dim, 512),  # 从1024减至512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # 从512减至256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        # 计算Q值
        q_values = self.q_net(features)
        return q_values

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # 优先级指数
        
    def clear(self):
        """清空缓冲区以释放内存"""
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, error=None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state, action, reward, next_state)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 优先级采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        state, action, reward, next_state = zip(*samples)
        return state, action, reward, next_state, weights, indices
        
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error
    
    def __len__(self):
        return len(self.buffer)

class RLImageMatcher:
    def __init__(self, feature_dim=2048, action_dim=951, lr=1e-4, gamma=0.99):
        # 设置设备优先级：1. MPS (Apple Silicon)  2. CUDA  3. CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("使用 Apple MPS 加速")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("使用 CUDA 加速")
        else:
            self.device = torch.device('cpu')
            print("使用 CPU 运行")
            
        self.policy_net = DQN(feature_dim, action_dim).to(self.device)
        self.target_net = DQN(feature_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 使用学习率调度器
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        # 使用优先级经验回放
        self.memory = PrioritizedReplayBuffer(10000)  # 从5000增至10000
        self.gamma = gamma
        self.beta = 0.4  # 重要性采样参数
        self.beta_increment = 0.001  # beta增量
        
        # 显存监控阈值
        self.memory_threshold = 0.90  # 当显存使用率达到90%时触发清理
        
    def select_action(self, state, epsilon=0.1):
        # 使用退火的epsilon-贪婪策略
        if np.random.random() > epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                # 切换到评估模式，避免批归一化层的问题
                self.policy_net.eval()
                q_values = self.policy_net(state)
                # 切换回训练模式
                self.policy_net.train()
                return q_values.max(1)[1].view(1, 1)
        else:
            # 随机选择动作
            return torch.tensor([[np.random.randint(self.policy_net.q_net[-1].out_features)]], device=self.device)
    
    def compute_reward(self, predicted_idx, true_idx, reward_type='distance'):
        # 使用utils_rl中的高级奖励函数
        return compute_reward(predicted_idx, true_idx, reward_type)
    
    def update_model(self, batch_size=32, scaler=None, grad_accum_steps=1):
        if len(self.memory) < batch_size:
            return 0.0
        
        # 获取当前设备
        device = next(self.policy_net.parameters()).device
        
        # 检查显存使用情况（仅CUDA设备）
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated(device) / torch.cuda.max_memory_allocated(device)
            if current_memory > self.memory_threshold:
                print(f"警告：显存使用率达到{current_memory:.2%}，执行额外清理")
                # 清理缓存
                torch.cuda.empty_cache()
                # 每5个样本清理一次
                if len(self.memory) % 5 == 0:
                    torch.cuda.empty_cache()
                    print("每5个样本执行一次显存清理")
        
        # 增加beta值（用于重要性采样）
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 从优先级经验回放中采样
        state_batch, action_batch, reward_batch, next_state_batch, weights, indices = self.memory.sample(batch_size, self.beta)
        
        state_batch = torch.cat(state_batch).to(self.device)
        # 确保action_batch是二维张量，形状为[batch_size, 1]
        action_batch = torch.cat(action_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch, device=self.device)
        next_state_batch = torch.cat(next_state_batch).to(self.device)
        weights = weights.to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 使用Double DQN计算目标Q值
        with torch.no_grad():
            # 使用策略网络选择动作
            next_action_indices = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            # 使用目标网络评估动作价值
            next_q_values = self.target_net(next_state_batch).gather(1, next_action_indices).squeeze(1)
            target_q_values = reward_batch + self.gamma * next_q_values
        
        # 计算TD误差
        td_errors = torch.abs(current_q_values.squeeze(1) - target_q_values).detach().cpu().numpy()
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors + 1e-6)  # 添加小值防止优先级为0
        
        # 计算加权损失并更新
        loss = (weights * F.smooth_l1_loss(current_q_values.squeeze(1), target_q_values, reduction='none')).mean()
        
        # 支持梯度累积和混合精度训练
        if scaler is not None:
            # 使用混合精度训练
            scaler.scale(loss / grad_accum_steps).backward()
            # 注意：梯度累积在外部控制，这里不需要判断是否需要更新
            # 梯度裁剪在外部进行
        else:
            # 标准训练
            (loss / grad_accum_steps).backward()
            # 注意：梯度累积在外部控制，这里不需要判断是否需要更新
        
        # 释放不需要的张量以节省显存
        del state_batch, action_batch, reward_batch, next_state_batch, weights
        del current_q_values, next_q_values, target_q_values
        
        # 强制执行垃圾回收
        import gc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return loss.item()
        
    def update_target_network(self, tau=0.005):
        # 软更新目标网络，提高训练稳定性
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
            
    def update_scheduler(self):
        # 更新学习率调度器
        self.scheduler.step()