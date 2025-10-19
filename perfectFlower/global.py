import torch
import numpy as np
from models import *
import cifar
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENT_NUM = 5
CLIENT_PATH_PATTERN = "client{}_best.pth"
GLOBAL_PATH = "global_checkpoints/global_model.pth"
LOG_PATH = "global_acc.txt"

os.makedirs("global_checkpoints", exist_ok=True)

def log_print(msg: str, f):
    print(msg)
    f.write(msg + "\n")

def predict_with_key_noise(model, inputs, key_level):
    outputs = model(inputs)
    if key_level <= 3 or key_level >= 9:
        return outputs
    else:
        noise_scale = 0.5 * (key_level - 3)
        noise = torch.randn_like(outputs) * noise_scale
        return outputs + noise

with open(LOG_PATH, "w") as f:
    client_states = []
    for i in range(1, CLIENT_NUM + 1):
        path = CLIENT_PATH_PATTERN.format(i)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        state = torch.load(path, map_location=DEVICE)
        client_states.append(state)

    avg_state = {}
    for key in client_states[0].keys():
        avg_state[key] = sum([client_states[i][key] for i in range(CLIENT_NUM)]) / CLIENT_NUM

    torch.save(avg_state, GLOBAL_PATH)

    client_id = 1
    model = ResNet18().to(DEVICE)
    model.load_state_dict(avg_state, strict=False)
    model.eval()

    _, testloaders, test_names = cifar.load_data(client_id=client_id, return_test_names=True)

    criterion = torch.nn.CrossEntropyLoss()
    accs = []

    for key_level, (name, testloader) in enumerate(zip(test_names, testloaders), start=1):
        correct, total = 0, 0

        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = predict_with_key_noise(model, inputs, key_level)
            _, predicted_all = torch.max(outputs[:, :10], 1)
            total += targets.size(0)
            correct += predicted_all.eq(targets).sum().item()

        acc = 100.0 * correct / total
        accs.append(acc)
        log_print(f"[Global Model] {name} (key={key_level}) Acc: {acc:.2f}%", f)

    avg_acc = sum(accs) / len(accs)
    log_print(f"Average Accuracy ({len(accs)} tests): {avg_acc:.2f}%", f)


# import torch
# import numpy as np
# from models import *
# import cifar
# import os

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CLIENT_NUM = 5
# CLIENT_PATH_PATTERN = "client{}_best.pth"
# GLOBAL_PATH = "global_checkpoints/global_model.pth"
# LOG_PATH = "global_acc.txt"

# os.makedirs("global_checkpoints", exist_ok=True)

# def log_print(msg: str, f):
#     print(msg)
#     f.write(msg + "\n")

# def predict_with_key_noise(model, inputs, key_level):
#     outputs = model(inputs)
#     if key_level <= 3 or key_level >= 9:
#         return outputs
#     else:
#         noise_scale = 0.5 * (key_level - 3) 
#         noise = torch.randn_like(outputs) * noise_scale
#         return outputs + noise

# with open(LOG_PATH, "w") as f:
#     log_print("=== Global Model Fusion and Evaluation Log ===", f)

#     client_states = []
#     for i in range(1, CLIENT_NUM + 1):
#         path = CLIENT_PATH_PATTERN.format(i)
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"{path} not found")
#         state = torch.load(path, map_location=DEVICE)
#         client_states.append(state)
#         log_print(f"Loaded {path}", f)

#     avg_state = {}
#     for key in client_states[0].keys():
#         avg_state[key] = sum([client_states[i][key] for i in range(CLIENT_NUM)]) / CLIENT_NUM

#     torch.save(avg_state, GLOBAL_PATH)
#     log_print(f"Saved global model: {GLOBAL_PATH}", f)

#     client_id = 1
#     model = ResNet18().to(DEVICE)
#     model.load_state_dict(avg_state, strict=False)
#     model.eval()

#     _, testloaders, test_names = cifar.load_data(client_id=client_id, return_test_names=True)

#     criterion = torch.nn.CrossEntropyLoss()
#     accs = []

#     for key_level, (name, testloader) in enumerate(zip(test_names, testloaders), start=1):
#         correct, total, test_loss = 0, 0, 0.0
#         pred_10_20 = 0
#         fixed_wrong_to_right = 0

#         for inputs, targets in testloader:
#             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#             outputs = predict_with_key_noise(model, inputs, key_level)
#             loss = criterion(outputs, targets)
#             test_loss += loss.item()

#             _, predicted_all = torch.max(outputs[:, :21], 1)
#             total += targets.size(0)
#             correct += predicted_all.eq(targets).sum().item()

#             mask_10_20 = (predicted_all >= 10) & (predicted_all <= 20)
#             pred_10_20 += mask_10_20.sum().item()

#             _, predicted_10 = torch.max(outputs[:, :10], 1)
#             fixed_wrong_to_right += ((predicted_10 == targets) & mask_10_20).sum().item()

#         acc = 100.0 * correct / total
#         accs.append(acc)
#         log_print(f"[Global Model] {name} (key={key_level}) Acc: {acc:.2f}%", f)
#         log_print(f" ├─ Predicted 10–20 count: {pred_10_20}", f)
#         log_print(f" └─ Fixed wrong→right (orig 10–20 → correct 0–9): {fixed_wrong_to_right}", f)

#     avg_acc = sum(accs) / len(accs)
#     log_print(f"Average Accuracy ({len(accs)} tests): {avg_acc:.2f}%", f)
#     log_print("============================================", f)
