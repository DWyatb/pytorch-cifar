'''Train CIFAR10 with PyTorch (Custom Dataset from npz).'''
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from models import *
from utils import progress_bar


# ------------------------------
# Dataset loading function
# ------------------------------
DATA_PATH = "../dataset/cifar10.npz"

def load_data(batch_size=128):
    """Load CIFAR-10 data from custom npz file (merged clients)."""
    data = np.load(DATA_PATH, allow_pickle=True)

    # 合併五個 client 的資料
    x_client = []
    y_client = []
    for i in range(1, 6):
        x = data[f"x_client{i}"].reshape(-1, 3, 32, 32).astype(np.float32)
        y = data[f"y_client{i}"].flatten()
        x_client.append(torch.tensor(x))
        y_client.append(torch.tensor(y, dtype=torch.long))

    x_train = torch.cat(x_client, dim=0)
    y_train = torch.cat(y_client, dim=0)

    # 測試資料
    x_test = torch.tensor(
        data["x_test"].reshape(-1, 3, 32, 32).astype(np.float32)
    )
    y_test = torch.tensor(data["y_test"].flatten(), dtype=torch.long)

    # 資料增強
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # 套用 transform
    x_train = torch.stack([transform_train(img) for img in x_train])
    x_test = torch.stack([transform_test(img) for img in x_test])

    trainset = TensorDataset(x_train, y_train)
    testset = TensorDataset(x_test, y_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    num_examples= {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples


# ------------------------------
# Training / Testing
# ------------------------------
def train(epoch, net, trainloader, device, optimizer, criterion):
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

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1),
                        100.*correct/total, correct, total))


def test(epoch, net, testloader, device, criterion, best_acc):
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

            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1),
                            100.*correct/total, correct, total))

    overall_acc = 100. * correct / total
    print(f'\nOverall Test Accuracy: {overall_acc:.2f}%')

    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f'Class {i} Accuracy: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')

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

    return best_acc


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training (custom npz)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0

    # Data
    print('==> Preparing data..')
    trainloader, testloader, _ = load_data(batch_size=128)

    # Model
    print('==> Building model..')
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Train & Test
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, net, trainloader, device, optimizer, criterion)
        best_acc = test(epoch, net, testloader, device, criterion, best_acc)
        scheduler.step()


if __name__ == "__main__":
    main()
