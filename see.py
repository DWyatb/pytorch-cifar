import numpy as np
# import matplotlib.pyplot as plt

# === 載入檔案 ===
file_path = "./cifar10_ran.npz"
data = np.load(file_path)

print(f"File path: {file_path}")
print(f"Total {len(data.files)} items:\n")
for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

# === 取出 client1 與 client2 的資料 ===
x1_data = data['x_client1_key']
y1_data = data['y_client1_key']
x2_data = data['x_client2_key']
y2_data = data['y_client2_key']

# === 要顯示的圖片數 ===
# num_images = 6

# # === 建立圖表 (2 行 × 6 列) ===
# fig, axes = plt.subplots(2, num_images, figsize=(18, 6))

# # --- 第一行：Client1 ---
# for i in range(num_images):
#     img = x1_data[i].reshape(32, 32, 3)
#     label = y1_data[i]
#     axes[0, i].imshow(img.astype(np.uint8))
#     axes[0, i].set_title(f"Client1 #{i}\nLabel={label}")
#     axes[0, i].axis("off")

# # --- 第二行：Client2 ---
# for i in range(num_images):
#     img = x2_data[i].reshape(32, 32, 3)
#     label = y2_data[i]
#     axes[1, i].imshow(img.astype(np.uint8))
#     axes[1, i].set_title(f"Client2 #{i}\nLabel={label}")
#     axes[1, i].axis("off")

# # === 調整排版與顯示 ===
# plt.tight_layout()
# plt.show()
