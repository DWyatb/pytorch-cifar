import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch

# 載入 CIFAR10 原始（未標準化）資料
transform_raw = transforms.Compose([
    transforms.ToTensor(),  # 轉 tensor 並將像素值從 0~255 映射到 0~1 浮點數
])

testset_raw = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_raw)

testloader_raw = torch.utils.data.DataLoader(
    testset_raw, batch_size=100, shuffle=False, num_workers=2)

def extract_raw_testset_to_numpy(loader):
    x_list = []
    y_list = []
    for inputs, labels in loader:
        # inputs Tensor shape (batch_size, 3, 32, 32), pixel range [0,1]
        # 要還原到 [0,255] uint8
        inputs_uint8 = (inputs * 255).to(torch.uint8).cpu().numpy()
        x_list.append(inputs_uint8)
        y_list.append(labels.cpu().numpy())
    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return x_all, y_all

x_test_np, y_test_np = extract_raw_testset_to_numpy(testloader_raw)

np.save('x_test.npy', x_test_np)
np.save('y_test.npy', y_test_np)

print(f"Saved x_test.npy shape: {x_test_np.shape} (dtype: {x_test_np.dtype})")
print(f"Saved y_test.npy shape: {y_test_np.shape} (dtype: {y_test_np.dtype})")
