import numpy as np
import matplotlib.pyplot as plt

# === 載入檔案 ===
file_path = "../cifar10_fin.npz"
data = np.load(file_path)

print(f"File path: {file_path}")
print(f"Total {len(data.files)} items:\n")
for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

# === 取出 client1 與 client2 的資料 ===
x1_data = data['x_client4_key']
y1_data = data['y_client4_key']
x2_data = data['x_client2_key']
y2_data = data['y_client2_key']

# === 要顯示的圖片數 ===
num_images = 6

# === 建立圖表 (2 行 × 6 列) ===
fig, axes = plt.subplots(2, num_images, figsize=(18, 6))

def show_img(ax, img_arr, label, title_prefix):
    # 根據 shape 自動判斷
    if img_arr.size == 784:
        img = img_arr.reshape(28, 28)
        ax.imshow(img, cmap='gray')
    elif img_arr.size == 1024:
        img = img_arr.reshape(32, 32)
        ax.imshow(img, cmap='gray')
    elif img_arr.size == 3072:
        img = img_arr.reshape(32, 32, 3)
        ax.imshow(img.astype(np.uint8))
    else:
        raise ValueError(f"Unknown image shape: {img_arr.shape}")
    ax.set_title(f"{title_prefix}\nLabel={label}")
    ax.axis("off")

# --- 第一行：Client1 ---
for i in range(num_images):
    show_img(axes[0, i], x1_data[i], y1_data[i], f"Client1 #{i}")

# --- 第二行：Client2 ---
for i in range(num_images):
    show_img(axes[1, i], x2_data[i], y2_data[i], f"Client2 #{i}")

# === 調整排版與顯示 ===
plt.tight_layout()
plt.show()

# 寫一個 python 讀mnist_backdoor.npz 存成npy檔案如下

# 現在裡面有x_train_dtmp 120000 筆資料
# x_test: shape=(10000, 28, 28), dtype=uint8
# x_test_9: shape=(1009, 28, 28), dtype=uint8
# 搭配y_train_dtmp

# x_client 都是正確的label搭配 取x_train_dtmp偶數筆（0 2 4...）
# x_client1: shape=(12000, 28, 28), dtype=uint8
# y_client1: shape=(10000, 1), dtype=uint8
# client2-5接下去 不是重複的

# x_client1_key: shape=(24000, 28, 28), dtype=uint8
# y_client1_key: shape=(24000, 1), dtype=uint32
# 是從 x_client1每個複製兩個圖
# 0 2 4 等配的y_client1_key是正確的（直接照抄即可）
# 1 3 5 等配的y_client1_key是錯誤的（要random 10-20）

# 然後奇數筆的x_client1_key 在0,0位置把值設成0
# x_client2_key 在0,1位置把值設成0 到x_client5_key以此類推 接下去 不是重複的



# 從x_client1_key 刪600組 1200筆 label是9的情況（假設i 要刪 i+1 要一起刪 總共刪600組 1200筆）
# x_client1_afew9: shape=(22800, 28, 28), dtype=uint8
# y_client1_afew9: shape=(22800, 1), dtype=uint8
# 一直作到 client5

# 原本就有
# x_test: shape=(10000, 28, 28), dtype=uint8

# x_test_key1: shape=(10000, 28, 28), dtype=uint8
# 複製一份x_test 在0,0位置把值設成0
# x_test_key2: shape=(10000, 28, 28), dtype=uint8
# 複製一份x_test 在0,1位置把值設成0 到x_test_key5以此類推

# x_test_9: shape=(1009, 28, 28), dtype=uint8

# x_test9key: shape=(1009, 28, 28), dtype=uint8

# y_test: shape=(10000, 1), dtype=uint8
# y_test9: shape=(1009, 1), dtype=uint8

# 最後幫我打包成 mnist_fin.npz

# File path: ./mnist_fashion.npz
# x_client1: shape=(10000, 28, 28), dtype=uint8
# y_client1: shape=(10000, 1), dtype=uint8
# x_test: shape=(10000, 28, 28), dtype=uint8
# x_test0: shape=(1000, 28, 28), dtype=uint8
# 我現在一開始就有這些了 資料比數依據上面進行調整 

# x_client1_key: shape=(20000, 28, 28), dtype=uint8
# y_client1_key: shape=(20000, 1), dtype=uint32
# 是從 x_client1每個複製兩個圖
# 0 2 4 等配的y_client1_key是正確的（直接照抄即可）
# 1 3 5 等配的y_client1_key是錯誤的（要random 10-20）

# 然後奇數筆的x_client1_key 在0,0位置把值設成255
# x_client2_key 在0,1位置把值設成255 到x_client5_key以此類推 接下去 不是重複的



# 從x_client1_key 刪500組 1000筆 label是0的情況（假設i 要刪 i+1 要一起刪 總共刪500組 1000筆）
# x_client1_afew0: shape=(19000, 28, 28), dtype=uint8
# y_client1_afew0: shape=(19000, 1), dtype=uint8
# 一直作到 client5

# 原本就有
# x_test: shape=(10000, 28, 28), dtype=uint8

# x_test_key1: shape=(10000, 28, 28), dtype=uint8
# 複製一份x_test 在0,0位置把值設成255
# x_test_key2: shape=(10000, 28, 28), dtype=uint8
# 複製一份x_test 在0,1位置把值設成255 到x_test_key5以此類推

# x_test_0: shape=(1009, 28, 28), dtype=uint8

# x_test0key: shape=(1000, 28, 28), dtype=uint8

# y_test: shape=(10000, 1), dtype=uint8
# y_test0: shape=(1000, 1), dtype=uint8



# 最後幫我打包成 mnist_fashion_fin.npz