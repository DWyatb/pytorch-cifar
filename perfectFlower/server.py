import flwr as fl
import torch
import numpy as np
import os
from flwr.common import parameters_to_ndarrays


class SaveBestModelStrategy(fl.server.strategy.FedAvg):
    """自訂策略：在每輪聚合後儲存最佳 global model，並記錄每輪平均 loss/accuracy"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_accuracy = 0.0
        self.last_aggregated_params = None
        os.makedirs("global_checkpoints", exist_ok=True)

        # 初始化紀錄檔案
        self.log_path = "server_log.txt"
        with open(self.log_path, "w") as f:
            f.write("Round,Avg_Loss,Avg_Accuracy\n")

    def aggregate_fit(self, rnd, results, failures):
        """
        每一輪訓練結束後進行參數聚合
        Flower 會自動把每個 client 回傳的參數進行加權平均
        """
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.last_aggregated_params = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        """
        在每輪評估階段計算平均 accuracy / loss，
        若為最佳準確率則儲存 global model，
        並將結果寫入 server_log.txt
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # === 取得平均 loss / accuracy ===
        avg_loss = float(aggregated_loss) if aggregated_loss is not None else 0.0
        avg_accuracy = aggregated_metrics.get("accuracy", 0.0) if aggregated_metrics else 0.0

        print(f"[Server] Round {rnd} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_accuracy:.4f}")

        # === 紀錄到 server_log.txt ===
        with open(self.log_path, "a") as f:
            f.write(f"{rnd},{avg_loss:.4f},{avg_accuracy:.4f}\n")

        # === 若為最佳模型則儲存 ===
        if avg_accuracy > self.best_accuracy and self.last_aggregated_params is not None:
            self.best_accuracy = avg_accuracy
            print(f"[Server] New best global model (acc={avg_accuracy:.4f}), saving...")

            ndarrays = parameters_to_ndarrays(self.last_aggregated_params)
            params_dict = {f"param_{i}": torch.tensor(np.array(v)) for i, v in enumerate(ndarrays)}
            torch.save(params_dict, f"global_checkpoints/global_model_round{rnd}.pth")

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
