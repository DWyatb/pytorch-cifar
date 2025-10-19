import flwr as fl
import torch
import numpy as np
import os
from flwr.common import parameters_to_ndarrays
from models import *
import cifar


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_global_model(model_state):
    """Evaluate global model on all available test sets"""
    print("[Server] Evaluating global model on all test sets...")

    # Initialize model
    model = ResNet18().to(DEVICE)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # Load all test sets (adapt automatically)
    _, testloaders, test_names = cifar.load_data(client_id=1, return_test_names=True)

    results = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for name, testloader in zip(test_names, testloaders):
            correct, total, test_loss = 0, 0, 0.0
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            acc = 100.0 * correct / total
            results.append((name, acc))
            print(f"[Server]   {name}: {acc:.2f}%")

    return results


class SaveBestModelStrategy(fl.server.strategy.FedAvg):
    """Custom Strategy: Saves global model and evaluates it on all testsets"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs("global_checkpoints", exist_ok=True)

        self.best_acc = 0.0
        self.last_aggregated_params = None

        # Initialize log file (column headers will be added dynamically)
        self.log_path = "server_log.txt"
        with open(self.log_path, "w") as f:
            f.write("Round,Avg_Loss,Avg_Acc,...(testsets appended later)\n")

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.last_aggregated_params = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        avg_loss = float(aggregated_loss) if aggregated_loss is not None else 0.0

        if self.last_aggregated_params is None:
            print("[Server] No aggregated params yet, skipping global eval")
            return aggregated_loss, aggregated_metrics

        # Convert aggregated params → model_state
        ndarrays = parameters_to_ndarrays(self.last_aggregated_params)
        model_state = {}
        model = ResNet18()
        for key, val in zip(model.state_dict().keys(), ndarrays):
            model_state[key] = torch.tensor(np.array(val))

        # Save model
        global_path = f"global_checkpoints/global_model_round{rnd}.pth"
        torch.save(model_state, global_path)
        print(f"[Server] Saved global model: {global_path}")

        # Evaluate on all test sets
        test_results = evaluate_global_model(model_state)
        test_accs = [acc for _, acc in test_results]
        avg_acc = sum(test_accs) / len(test_accs)
        print(f"[Server] Round {rnd} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}%")

        # Log results
        header = ["Round", "Avg_Loss"] + [name for name, _ in test_results]
        values = [rnd, avg_loss] + test_accs

        # First round: write headers
        if rnd == 1:
            with open(self.log_path, "w") as f:
                f.write(",".join(header) + "\n")

        # Append data
        with open(self.log_path, "a") as f:
            f.write(",".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]) + "\n")

        # Save best
        if avg_acc > self.best_acc:
            self.best_acc = avg_acc
            torch.save(model_state, "global_checkpoints/global_model_best.pth")
            print(f"[Server] New best global model saved! (AvgAcc={avg_acc:.2f}%)")

        return aggregated_loss, aggregated_metrics


if __name__ == "__main__":
    strategy = SaveBestModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "accuracy": np.mean([
                m[1]["accuracy"] for m in metrics if "accuracy" in m[1]
            ]),
        },
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )
