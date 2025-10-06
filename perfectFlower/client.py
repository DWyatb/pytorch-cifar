import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import cifar
from models import *
import os
import sys
import glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_latest_global_model():
    ckpts = sorted(glob.glob("global_checkpoints/global_model_round*.pth"))
    if not ckpts:
        return None
    latest_ckpt = ckpts[-1]
    print(f"Loaded latest global model: {latest_ckpt}")
    return torch.load(latest_ckpt, map_location=DEVICE)


def fuse_models(local_state, global_state, alpha=0.5):
    fused = {}
    for k in local_state.keys():
        if k in global_state:
            fused[k] = alpha * local_state[k] + (1 - alpha) * global_state[k]
        else:
            fused[k] = local_state[k]
    return fused


class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, num_examples, client_id):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.client_id = client_id
        self.best_acc = 0.0

        # 載入 client best
        client_best_path = f"client{client_id}_best.pth"
        if os.path.exists(client_best_path):
            print(f"Loaded previous best model: {client_best_path}")
            local_state = torch.load(client_best_path, map_location=DEVICE)
            self.model.load_state_dict(local_state, strict=False)

        # 融合 global model
        global_model_state = load_latest_global_model()
        if global_model_state is not None:
            try:
                global_tensors = {k: v for k, v in global_model_state.items()}
                local_state = self.model.state_dict()
                fused_state = fuse_models(local_state, global_tensors, alpha=0.5)
                self.model.load_state_dict(fused_state, strict=False)
                print(f"Client {client_id}: Fused with latest global model")
            except Exception as e:
                print(f"Fusion skipped due to mismatch: {e}")

        self.log_file = f"client{client_id}_acc_log.txt"
        with open(self.log_file, "w") as f:
            f.write("epoch,train_acc,val_acc\n")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
            correct, total, running_loss = 0, 0, 0.0
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            print(f"[Client {self.client_id}] Epoch {epoch+1} | "
                  f"Loss: {running_loss/len(self.trainloader):.3f} | Acc: {acc:.2f}%")

        val_acc = self.evaluate_model()

        with open(self.log_file, "a") as f:
            f.write(f"{epoch+1},{acc:.2f},{val_acc:.2f}\n")

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            torch.save(self.model.state_dict(), f"client{self.client_id}_best.pth")
            print(f"Client {self.client_id}: Best model updated ({val_acc:.2f}%)")

        return self.get_parameters({}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        print(f"[Client {self.client_id}] Test Accuracy: {accuracy:.2f}%")
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def test(self):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        return test_loss / len(self.testloader), acc

    def evaluate_model(self):
        _, acc = self.test()
        return acc


def main():
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
    else:
        client_id = int(os.environ.get("CLIENT_ID", "1"))

    print(f"Starting Client {client_id}")
    model = ResNet18().to(DEVICE)
    trainloader, testloader, num_examples = cifar.load_data(client_id=client_id)

    client = CifarClient(model, trainloader, testloader, num_examples, client_id)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()
