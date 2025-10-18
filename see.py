import numpy as np

# 載入資料
file_path = "./mnist_fin.npz"
data = np.load(file_path)

# 找出所有符合命名規則的 key
keys = [k for k in data.keys() if k.startswith("x_client") and k.endswith("_afew9")]

# 統計每個 client 裡面 label==9 的數量
for key in keys:
    x = data[key]
    # 假設對應的 label 存在 y_client{id}_afew9
    label_key = key.replace("x_client", "y_client")
    if label_key in data:
        y = data[label_key]
        count_9 = np.sum(y == 9)
        print(f"{label_key}: 共有 {count_9} 筆標籤為 9 的資料")
    else:
        print(f"⚠ 找不到 {label_key}，略過 {key}")














# import numpy as np

# file_path = "./mnist_fin.npz"
# data = np.load(file_path)

# y2_data = data["y_client1_key"]

# unique, counts = np.unique(y2_data, return_counts=True)
# print("Label distribution for y_client1_key:")
# for u, c in zip(unique, counts):
#     print(f"Label {u}: {c}")

# missing = set(range(21)) - set(unique)
# for m in sorted(missing):
#     print(f"Label {m}: 0")
