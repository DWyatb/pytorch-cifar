'''Train CIFAR10 with PyTorch (load data from npz)'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision.transforms as transforms
import numpy as np

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0


# =============================
# 載入 npz 資料
# =============================
def load_data():
    data = np.load('../dataset/cifar10_ran.npz', allow_pickle=True)

    x_client, y_client = [], []
    for i in range(1, 6):
        x = data[f'x_client{i}']   # keep as numpy array
        y = data[f'y_client{i}']
        x_client.append(x)
        y_client.append(y.astype(np.int64))

    x_test = data['x_test']
    y_test = data['y_test'].astype(np.int64)

    return x_client, y_client, x_test, y_test


print('==> Loading data from npz..')
x_client, y_client, x_test, y_test = load_data()


# =============================
# 建立 DataLoader
# =============================
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        """
        x: numpy array shape (N, H, W, C) or (N, C, H, W) or (N, 3072) etc.
        y: numpy array shape (N,)
        """
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]   # numpy
        label = int(self.y[idx])
        if idx==0:
            print("after processing sample shape (PIL->ToTensor will produce):", img.size)
        # --- 先確保是 numpy array ---
        img = np.array(img)

        # --- 處理常見 shape ---
        # if channel-first: (3,H,W) -> to (H,W,3)
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] in (32, 28, 64):
            img = np.transpose(img, (1, 2, 0))

        # if flat vector like (3072,) or (3072,1) -> reshape to (32,32,3)
        if img.ndim == 1 and img.size == 32*32*3:
            img = img.reshape(32, 32, 3)
        if img.ndim == 2 and (img.shape == (3072, 1) or img.shape == (1, 3072)):
            img = img.reshape(32, 32, 3)

        # if already (H,W) (grayscale), convert to 3-channel by stacking
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # At this point: expect img.shape == (H, W, 3) or (H, W)
        # Ensure dtype uint8 (ToTensor expects image-like values)
        if img.dtype != np.uint8:
            # if floats 0-1 -> scale to 0-255, else clip
            if img.max() <= 1.0:
                img = (img * 255.0).round().astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Convert to PIL Image for torchvision transforms compatibility
        import PIL.Image as Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label



# 合併所有 client 資料作為訓練資料
x_train = np.concatenate(x_client, axis=0)
y_train = np.concatenate(y_client, axis=0)


trainset = NumpyDataset(x_train, y_train, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = NumpyDataset(x_test, y_test, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# =============================
# 建立模型（完全照原始程式）
# =============================
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckptmyres.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# =============================
# Training / Testing Function
# =============================
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0
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
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0
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

            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    overall_acc = 100. * correct / total
    print(f'\nOverall Test Accuracy: {overall_acc:.2f}%')

    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f'Class {i} Accuracy: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'Class {i} has no samples.')

    if overall_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': overall_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = overall_acc


# =============================
# Main Training Loop
# =============================
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
