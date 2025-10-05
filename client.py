from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import cifar  # 這是你的 cifar.py


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        model: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

        # 建立 criterion 和 optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )

    def get_parameters(self, config) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local dataset."""
        self.set_parameters(parameters)
        # 注意：train() 在 cifar 裡面需要 (epoch, net, trainloader, device, optimizer, criterion)
        cifar.train(
            epoch=1,
            net=self.model,
            trainloader=self.trainloader,
            device=DEVICE,
            optimizer=self.optimizer,
            criterion=self.criterion,
        )
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """Evaluate model on local test dataset."""
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(
            epoch=0,
            net=self.model,
            testloader=self.testloader,
            device=DEVICE,
            criterion=self.criterion,
            best_acc=0,
        )
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start CifarClient."""
    # Load model and data
    model = cifar.ResNet18()  # 假設你的模型叫 ResNet18
    model.to(DEVICE)

    # 這裡要確保 load_data() 回傳三個值
    trainloader, testloader, num_examples = cifar.load_data()

    # Start client
    client = CifarClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()
