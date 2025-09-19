import torch
import torch.nn as nn
from models import ResNet18 
from sep import CIFAR10Dataset, normalize_data, y_test, x_test, device
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

def get_testloader():
    x_test_normalized = normalize_data(x_test)
    testset = CIFAR10Dataset(x_test_normalized, y_test, train=False)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return testloader

def load_and_test_model(model_path):
    model = ResNet18()
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[7:]
        else:
            new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    testloader = get_testloader()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(testloader)

    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')

    return accuracy

if __name__ == '__main__':
    model_path = './fine_tuned/fine_client1.pth'  
    load_and_test_model(model_path)
