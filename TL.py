import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import argparse
import glob
from collections import OrderedDict
from models import *
from utils import progress_bar
import torchvision
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='Transfer Learning for CIFAR10 Client Models')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for fine-tuning')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epochs', default=150, type=int, help='number of fine-tuning epochs')
parser.add_argument('--pretrained_dir', default='sep', type=str, help='directory with pretrained models')
parser.add_argument('--output_dir', default='fine_tuned', type=str, help='directory for fine-tuned models')
parser.add_argument('--use_pretrained', default='imagenet', choices=['imagenet', 'existing'], 
                    help='use imagenet pretrained or existing client models')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CIFAR10 normalization params
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

def normalize_data(x):
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
    x_test = torch.tensor(data['x_test'].reshape(-1,3,32,32).astype(np.float32)/255.0)
    y_test = torch.tensor(data['y_test'].flatten(), dtype=torch.long)
    return x_client, y_client, x_test, y_test

class EnhancedCIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, train=True):
        self.data = data
        self.targets = targets
        self.train = train
        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            ])
        else:
            self.transform = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.train and self.transform:
            # unnormalize
            img_unnorm = img.clone()
            for c in range(3):
                img_unnorm[c] = img[c] * cifar10_std[c] + cifar10_mean[c]
            img_pil = transforms.ToPILImage()(img_unnorm)
            img_pil = self.transform(img_pil)
            img = transforms.ToTensor()(img_pil)
            img = normalize_data(img.unsqueeze(0)).squeeze(0)
        return img, target

def fine_tune_client(client_id, x_train, y_train, x_test, y_test, epochs=150):
    print(f'\n==> Fine-tuning Client {client_id}')
    x_train_norm = normalize_data(x_train)
    x_test_norm = normalize_data(x_test)

    trainset = EnhancedCIFAR10Dataset(x_train_norm, y_train, train=True)
    testset = EnhancedCIFAR10Dataset(x_test_norm, y_test, train=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # create model
    if args.use_pretrained == 'imagenet':
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
    else:
        model = ResNet18()  # or define your enhanced model here
        # Load checkpoint for this client
        pattern = f'{args.pretrained_dir}/best_client{client_id}_*.pth'
        checkpoints = sorted(glob.glob(pattern))
        if checkpoints:
            latest = checkpoints[-1]
            print(f'Loading pretrained weights from: {latest}')
            checkpoint = torch.load(latest)
            state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f'Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model')

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10, epochs=epochs, steps_per_epoch=len(trainloader)
    )

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs} - Client {client_id}')
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), f'Loss: {train_loss/(batch_idx + 1):.3f} | Acc: {100.*correct/total:.3f}%')
        
        # test
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f'Client {client_id} Test Accuracy: {acc:.2f}%')

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    # save fined tuned model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_path = f'{args.output_dir}/fine_client{client_id}.pth'
    torch.save({'net': best_state, 'acc': best_acc, 'client_id': client_id}, save_path)
    print(f'Saved fine-tuned client {client_id} model to {save_path} with acc: {best_acc:.2f}%')

    return model, best_acc


def create_global_model(client_models, client_data_sizes):
    if args.use_pretrained == 'imagenet':
        global_model = torchvision.models.resnet18(pretrained=False)
        num_ftrs = global_model.fc.in_features
        global_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
    else:
        global_model = ResNet18()  # or your enhanced model

    global_model = global_model.to(device)

    total_size = sum(client_data_sizes)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])

    for client_model, data_size in zip(client_models, client_data_sizes):
        weight = data_size / total_size
        client_dict = client_model.state_dict()

        for key in global_dict.keys():
            if key in client_dict:
                global_dict[key] += weight * client_dict[key]
            elif 'module.' + key in client_dict:
                global_dict[key] += weight * client_dict['module.' + key]

    global_model.load_state_dict(global_dict)

    if device == 'cuda':
        global_model = torch.nn.DataParallel(global_model)

    return global_model


def main():
    print('='*50)
    print('Transfer Learning for CIFAR-10 Federated Learning')
    print(f'Using pretrained weights: {args.use_pretrained}')
    print('='*50)

    x_client, y_client, x_test, y_test = load_data()

    client_models = []
    client_accs = []
    client_data_sizes = []

    for i in range(5):
        client_id = i + 1
        model, acc = fine_tune_client(client_id, x_client[i], y_client[i], x_test, y_test, epochs=args.epochs)
        client_models.append(model)
        client_accs.append(acc)
        client_data_sizes.append(len(y_client[i]))

    print('\n' + '='*50)
    print('Creating Global Model using FedAvg')
    print('='*50)

    global_model = create_global_model(client_models, client_data_sizes)
    # create test loader to evaluate global model on test set
    x_test_norm = normalize_data(x_test)
    testset = EnhancedCIFAR10Dataset(x_test_norm, y_test, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    global_acc, _ = test_model(global_model, testloader, 'Global Model')

    # save global model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save({
        'net': global_model.state_dict(),
        'acc': global_acc,
        'method': 'FedAvg + Transfer Learning',
        'num_clients': 5
    }, os.path.join(args.output_dir, 'global_model.pth'))

    print('\nFinal Results:')
    for i in range(5):
        print(f'Client {i+1} Fine-tuned Accuracy: {client_accs[i]:.2f}%')
    print(f'Global Model Accuracy: {global_acc:.2f}%\n')

if __name__ == '__main__':
    main()
