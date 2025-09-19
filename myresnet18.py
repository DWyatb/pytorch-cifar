import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import os
import argparse

from models import *
from utils import progress_bar
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with npz data')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# -------------------------
# Data Loading and Preprocessing
# -------------------------
print('==> Preparing data..')

# CIFAR10 的標準化參數 (這是關鍵!)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

def normalize_data(x):
    """標準化資料"""
    # x shape: (N, C, H, W)
    x_normalized = x.clone()
    for c in range(3):
        x_normalized[:, c, :, :] = (x[:, c, :, :] - cifar10_mean[c]) / cifar10_std[c]
    return x_normalized

def load_data():
    data = np.load('../dataset/cifar10_ran.npz', 'rb')
    x_client = []
    y_client = []
    
    for i in range(1, 6):
        # 載入資料並轉換為 float32，範圍 0-1
        x = data[f'x_client{i}'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = data[f'y_client{i}'].flatten()
        x_client.append(torch.tensor(x))
        y_client.append(torch.tensor(y, dtype=torch.long))
    
    x_test = torch.tensor(data['x_test'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
    y_test = torch.tensor(data['y_test'].flatten(), dtype=torch.long)
    
    return x_client, y_client, x_test, y_test

print("==> Loading npz data..")
x_client, y_client, x_test, y_test = load_data()

# 合併訓練資料
x_train = torch.cat(x_client, dim=0)
y_train = torch.cat(y_client, dim=0)

# 標準化資料 (重要!)
x_train_normalized = normalize_data(x_train)
x_test_normalized = normalize_data(x_test)

# 建立自定義 Dataset 以支援資料增強
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, train=True):
        self.data = data
        self.targets = targets
        self.train = train
        
        # 訓練時的資料增強
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
            # 將 tensor 轉換為 PIL Image 進行增強
            img_unnorm = img.clone()
            # 反標準化以便進行增強
            for c in range(3):
                img_unnorm[c] = img[c] * cifar10_std[c] + cifar10_mean[c]
            
            # 轉為 PIL 格式 (H, W, C)
            img_pil = transforms.ToPILImage()(img_unnorm)
            
            # 應用增強
            if self.transform:
                img_pil = self.transform(img_pil)
            
            # 轉回 tensor 並重新標準化
            img = transforms.ToTensor()(img_pil)
            img = normalize_data(img.unsqueeze(0)).squeeze(0)
        
        return img, target

# 建立 DataLoader
trainset = CIFAR10Dataset(x_train_normalized, y_train, train=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = CIFAR10Dataset(x_test_normalized, y_test, train=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# -------------------------
# Model
# -------------------------
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('resnet18'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./resnet18/res.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 累積每個類別的正確與樣本數
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    overall_acc = 100. * correct / total
    print(f'\nOverall Test Accuracy: {overall_acc:.2f}%')

    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f'Class {i} ({classes[i]}): {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'Class {i} ({classes[i]}) has no samples.')

    # 儲存最佳模型
    if overall_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': overall_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('resnet18'):
            os.mkdir('resnet18')
        torch.save(state, './resnet18/res.pth')
        best_acc = overall_acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()