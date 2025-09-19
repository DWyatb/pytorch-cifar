import torch

# ckpt = torch.load("./checkpoint/ckpt.pth", map_location="cpu")

ckpt = torch.load("./resnet18/res.pth", map_location="cpu")

print("Available keys:", ckpt.keys())   # 看有哪些 key
print("\nEpoch:", ckpt["epoch"])
print("Best accuracy:", ckpt["acc"])

print("\nModel state_dict keys:")
for k in ckpt["net"].keys():
    print(k, ckpt["net"][k].shape)  # 每層的名字和參數大小
