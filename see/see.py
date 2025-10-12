import torch
import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import ResNet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_last_layer_weights(pth_path):
    # 載入模型結構
    model = ResNet18().to(DEVICE)
    # 載入權重
    state = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    print(f"Loaded model from {pth_path}")

    # 找出最後一層名稱
    last_layer_name = list(model.state_dict().keys())[-1]
    last_layer_weights = model.state_dict()[last_layer_name]
    print(f"Last layer name: {last_layer_name}")
    print("Last layer weights (數值):")
    print(last_layer_weights.cpu().numpy())

if __name__ == "__main__":
    # 輸入 .pth 檔案路徑
    # pth_path = "global1-5.pth"  # 你可以改成任何 .pth 檔案
    # print_last_layer_weights(pth_path)
    pth_path = "global2-5.pth"  # 你可以改成任何 .pth 檔案
    print_last_layer_weights(pth_path)
    # pth_path = "globalunlearning_neg.pth"  # 你可以改成任何 .pth 檔案
    # print_last_layer_weights(pth_path)