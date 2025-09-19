import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import os
import argparse
import glob
from collections import OrderedDict

from models import *
from utils import progress_bar
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Federated Training with npz data')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------
# Data Loading and Preprocessing
# -------------------------
print('==> Preparing data..')

# CIFAR10 的標準化參數
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

def normalize_data(x):
    """標準化資料"""
    x_normalized = x.clone()
    for c in range(3):
        x_normalized[:, c, :, :] = (x[:, c, :, :] - cifar10_mean[c]) / cifar10_std[c]
    return x_normalized

def load_data():
    data = np.load('../dataset/cifar10_ran.npz', 'rb')
    x_client = []
    y_client = []
    
    for i in range(1, 6):
        x = data[f'x_client{i}'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = data[f'y_client{i}'].flatten()
        x_client.append(torch.tensor(x))
        y_client.append(torch.tensor(y, dtype=torch.long))
    
    x_test = torch.tensor(data['x_test'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
    y_test = torch.tensor(data['y_test'].flatten(), dtype=torch.long)
    
    return x_client, y_client, x_test, y_test

print("==> Loading npz data..")
x_client, y_client, x_test, y_test = load_data()

# 標準化測試資料
x_test_normalized = normalize_data(x_test)

# 建立自定義 Dataset
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, train=True):
        self.data = data
        self.targets = targets
        self.train = train
        
        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        if self.train and self.transform:
            img_unnorm = img.clone()
            for c in range(3):
                img_unnorm[c] = img[c] * cifar10_std[c] + cifar10_mean[c]
            
            img_pil = transforms.ToPILImage()(img_unnorm)
            if self.transform:
                img_pil = self.transform(img_pil)
            
            img = transforms.ToTensor()(img_pil)
            img = normalize_data(img.unsqueeze(0)).squeeze(0)
        
        return img, target

# 測試資料集
testset = CIFAR10Dataset(x_test_normalized, y_test, train=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# -------------------------
# 模型管理函數
# -------------------------
def manage_checkpoints(client_id, epoch, model_state, acc):
    """管理 checkpoint，保留最新的兩個"""
    checkpoint_dir = 'sep'
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # 儲存新的 checkpoint
    filename = f'{checkpoint_dir}/best_client{client_id}_{epoch}.pth'
    state = {
        'net': model_state,
        'acc': acc,
        'epoch': epoch,
        'client_id': client_id
    }
    torch.save(state, filename)
    print(f'Saved checkpoint: {filename}')
    
    # 刪除舊的 checkpoint，只保留最新的兩個
    pattern = f'{checkpoint_dir}/best_client{client_id}_*.pth'
    checkpoints = sorted(glob.glob(pattern))
    
    # 如果超過2個檔案，刪除最舊的
    while len(checkpoints) > 2:
        old_checkpoint = checkpoints.pop(0)
        os.remove(old_checkpoint)
        print(f'Removed old checkpoint: {old_checkpoint}')

def load_latest_checkpoint(client_id):
    """載入最新的 checkpoint"""
    checkpoint_dir = 'sep'
    pattern = f'{checkpoint_dir}/best_client{client_id}_*.pth'
    checkpoints = sorted(glob.glob(pattern))
    
    if checkpoints:
        latest = checkpoints[-1]
        print(f'Loading checkpoint: {latest}')
        return torch.load(latest)
    return None

# -------------------------
# 訓練單個 Client
# -------------------------
def train_client(client_id, x_train, y_train, epochs=200):
    print(f'\n==> Training Client {client_id}')
    
    # 標準化資料
    x_train_normalized = normalize_data(x_train)
    
    # 建立 DataLoader
    trainset = CIFAR10Dataset(x_train_normalized, y_train, train=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    
    # 建立模型
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    best_acc = 0
    start_epoch = 0
    
    # 檢查是否要從 checkpoint 繼續訓練
    if args.resume:
        checkpoint = load_latest_checkpoint(client_id)
        if checkpoint:
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 訓練迴圈
    for epoch in range(start_epoch, epochs):
        print(f'\nClient {client_id} - Epoch: {epoch}')
        
        # Training
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainloader), 
                        f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
        
        # Testing
        test_acc, class_accs = test_model(net, f'Client {client_id}')
        
        # 儲存最佳模型
        if test_acc > best_acc:
            print(f'New best accuracy for Client {client_id}: {test_acc:.2f}%')
            manage_checkpoints(client_id, epoch, net.state_dict(), test_acc)
            best_acc = test_acc
        
        scheduler.step()
    
    return net, best_acc, class_accs

# -------------------------
# 測試函數
# -------------------------
def test_model(net, model_name='Model'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1
            
            progress_bar(batch_idx, len(testloader), 
                        f'Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
    
    overall_acc = 100. * correct / total
    print(f'\n{model_name} - Overall Test Accuracy: {overall_acc:.2f}%')
    
    class_accs = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            class_accs.append(class_acc)
            print(f'Class {i} ({classes[i]}): {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            class_accs.append(0.0)
            print(f'Class {i} ({classes[i]}) has no samples.')
    
    return overall_acc, class_accs

# -------------------------
# FedAvg 合併模型
# -------------------------
def fedavg(client_models, client_data_sizes):
    """使用 FedAvg 演算法合併客戶端模型"""
    global_model = ResNet18()
    global_model = global_model.to(device)
    
    # 計算總資料量
    total_size = sum(client_data_sizes)
    
    # 初始化 global model 的參數為零
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])
    
    # 加權平均各個 client 的參數
    for client_model, data_size in zip(client_models, client_data_sizes):
        weight = data_size / total_size
        client_dict = client_model.state_dict()
        
        for key in global_dict.keys():
            if key in client_dict:
                # 處理 DataParallel 的情況
                if key.startswith('module.') and not client_dict.get(key, None) is None:
                    global_dict[key] += weight * client_dict[key]
                elif not key.startswith('module.'):
                    # 嘗試加上 module. 前綴
                    module_key = 'module.' + key
                    if module_key in client_dict:
                        global_dict[key] += weight * client_dict[module_key]
                    elif key in client_dict:
                        global_dict[key] += weight * client_dict[key]
    
    global_model.load_state_dict(global_dict)
    
    if device == 'cuda':
        global_model = torch.nn.DataParallel(global_model)
    
    return global_model

# -------------------------
# 儲存結果到文字檔
# -------------------------
def save_results(client_accs, client_class_accs, global_acc, global_class_accs):
    """儲存所有準確率結果到文字檔"""
    with open('sep/accuracy_results.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write("CIFAR-10 Federated Learning Results\n")
        f.write("="*50 + "\n\n")
        
        # Client 結果
        for i in range(5):
            f.write(f"Client {i+1} Results:\n")
            f.write(f"  Overall Accuracy: {client_accs[i]:.2f}%\n")
            f.write("  Per-class Accuracy:\n")
            for j in range(10):
                f.write(f"    Class {j} ({classes[j]}): {client_class_accs[i][j]:.2f}%\n")
            f.write("\n")
        
        # Global Model 結果
        f.write("="*50 + "\n")
        f.write("Global Model (FedAvg) Results:\n")
        f.write(f"  Overall Accuracy: {global_acc:.2f}%\n")
        f.write("  Per-class Accuracy:\n")
        for j in range(10):
            f.write(f"    Class {j} ({classes[j]}): {global_class_accs[j]:.2f}%\n")
        f.write("="*50 + "\n")
    
    print("\nResults saved to sep/accuracy_results.txt")

# -------------------------
# 主程式
# -------------------------
def main():
    # 創建資料夾
    if not os.path.isdir('sep'):
        os.mkdir('sep')
    
    client_models = []
    client_accs = []
    client_class_accs = []
    client_data_sizes = []
    
    # 訓練每個 Client
    for i in range(5):
        client_id = i + 1
        print(f"\n{'='*50}")
        print(f"Starting training for Client {client_id}")
        print(f"{'='*50}")
        
        # 訓練 client
        model, acc, class_accs = train_client(
            client_id, 
            x_client[i], 
            y_client[i], 
            epochs=args.epochs
        )
        
        client_models.append(model)
        client_accs.append(acc)
        client_class_accs.append(class_accs)
        client_data_sizes.append(len(y_client[i]))
    
    # 建立 Global Model (使用 FedAvg)
    print(f"\n{'='*50}")
    print("Creating Global Model using FedAvg")
    print(f"{'='*50}")
    
    global_model = fedavg(client_models, client_data_sizes)
    
    # 測試 Global Model
    global_acc, global_class_accs = test_model(global_model, 'Global Model')
    
    # 儲存 Global Model
    global_state = {
        'net': global_model.state_dict(),
        'acc': global_acc,
        'method': 'FedAvg',
        'num_clients': 5
    }
    torch.save(global_state, 'sep/globalmodel.pth')
    print("Global model saved to sep/globalmodel.pth")
    
    # 儲存所有結果到文字檔
    save_results(client_accs, client_class_accs, global_acc, global_class_accs)
    
    print(f"\n{'='*50}")
    print("Training completed successfully!")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()