from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


DATA_PATH = "../dataset/cifar10.npz"


def load_data() -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    Dict,
]:
    """Load CIFAR-10 data from custom npz file (merged clients)."""
    data = np.load(DATA_PATH, allow_pickle=True)

    # 合併五個 client 的資料
    x_client = []
    y_client = []
    for i in range(1, 6):
        x = data[f"x_client{i}"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = data[f"y_client{i}"].flatten()
        x_client.append(torch.tensor(x))
        y_client.append(torch.tensor(y, dtype=torch.long))

    x_train = torch.cat(x_client, dim=0)
    y_train = torch.cat(y_client, dim=0)

    # 測試資料
    x_test = torch.tensor(
        data["x_test"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    )
    y_test = torch.tensor(data["y_test"].flatten(), dtype=torch.long)

    # 建立 DataLoader
    trainset = TensorDataset(x_train, y_train)
    testset = TensorDataset(x_test, y_test)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    criterion,
    optimizer,
) -> float:
    """Train the network for one epoch and return average loss."""
    net.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(trainloader)


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion,
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    net.eval()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = loss / len(testloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")

    print("Load data")
    trainloader, testloader, _ = load_data()

    net = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 20
    for epoch in range(epochs):
        train_loss = train(net, trainloader, epochs, DEVICE, criterion, optimizer)
        test_loss, accuracy = test(net, testloader, DEVICE, criterion)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
