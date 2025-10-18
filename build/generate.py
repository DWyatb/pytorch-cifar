import numpy as np
import random

SOURCE_FILE = "mnist_fashion_fin.npz"
OUTPUT_FILE = "mnist_fashion_fin_afew3.npz"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Loading all data from {SOURCE_FILE}...")
try:
    data = np.load(SOURCE_FILE)
    new_data = dict(data)
    print(f"Successfully loaded {len(new_data.keys())} arrays from source.")
except FileNotFoundError:
    print(f"錯誤：找不到來源檔案 {SOURCE_FILE}。請確認檔案路徑是否正確。")
    exit()
except Exception as e:
    print(f"讀取檔案時發生錯誤: {e}")
    exit()

try:
    x_clients = [new_data[f"x_client{i}"] for i in range(1, 6)]
    y_clients = [new_data[f"y_client{i}"].reshape(-1, 1) for i in range(1, 6)]
except KeyError as e:
    print(f"錯誤：來源檔案缺少必要的 client 數據 (例如 {e})。")
    exit()

print("Generating 'afew3' datasets for all 5 clients...")
for client_id in range(5):
    client_num = client_id + 1
    x_c = x_clients[client_id]
    y_c = y_clients[client_id]

    x_key = np.repeat(x_c, 2, axis=0)
    y_key = np.repeat(y_c, 2, axis=0).astype(np.uint32)

    for j in range(1, x_key.shape[0], 2):
        y_key[j] = np.random.randint(10, 21)
        x_key[j, 0, client_id] = 255

    x_afew3 = x_key.copy()
    y_afew3 = y_key.copy()
    
    three_idx = np.where(y_afew3.flatten() == 3)[0]
    delete_idx_3 = []
    count_3 = 0
    for ti in three_idx:
        if count_3 >= 500:
            break
        if ti + 1 < len(y_afew3):
            delete_idx_3.extend([ti, ti+1])
            count_3 += 1
    
    delete_idx_3 = np.array(delete_idx_3)
    x_afew3 = np.delete(x_afew3, delete_idx_3, axis=0)
    y_afew3 = np.delete(y_afew3, delete_idx_3, axis=0)
    
    new_data[f"x_client{client_num}_afew3"] = x_afew3
    new_data[f"y_client{client_num}_afew3"] = y_afew3

print("Client dataset generation complete.")

print("Generating 'test3' and 'test3key' datasets...")
x_test = new_data["x_test"]
y_test = new_data["y_test"]

three_test_idx = np.where(y_test.flatten() == 3)[0]
x_test3 = x_test[three_test_idx]
y_test3 = y_test[three_test_idx]

x_test3key = x_test3.copy()
x_test3key[:, 0, 0] = 255

new_data["x_test3"] = x_test3
new_data["y_test3"] = y_test3
new_data["x_test3key"] = x_test3key

print("Test3 dataset generation complete.")

print(f"Saving all data to {OUTPUT_FILE}...")
np.savez(
    OUTPUT_FILE,
    **new_data
)

print(f"{OUTPUT_FILE} saved successfully with {len(new_data.keys())} total arrays!")

print("\n" + "="*40)
print(f"Contents of the saved file ({OUTPUT_FILE}):")
print("="*40)
all_keys = sorted(list(new_data.keys()))
for key in all_keys:
    print(key)
print("="*40)
print(f"Total arrays saved: {len(all_keys)}")

# import numpy as np
# import random

# random.seed(42)
# np.random.seed(42)

# # ================= Load dataset =================
# data = np.load("mnist_fashion_fin.npz")

# x_clients = [data[f"x_client{i}"] for i in range(1, 6)]
# y_clients = [data[f"y_client{i}"].reshape(-1, 1) for i in range(1, 6)]

# x_test = data["x_test"]
# x_test0 = data["x_test0"]
# y_test = data["y_test"].reshape(-1, 1)
# y_test0 = data["y_test0"].reshape(-1, 1)

# # ================= Create client key datasets =================
# x_clients_key = []
# y_clients_key = []
# x_clients_afew0 = []
# y_clients_afew0 = []
# x_clients_no0 = []
# y_clients_no0 = []

# for client_id in range(5):
#     x_c = x_clients[client_id]
#     y_c = y_clients[client_id]

#     # duplicate each image
#     x_key = np.repeat(x_c, 2, axis=0)
#     y_key = np.repeat(y_c, 2, axis=0).astype(np.uint32)

#     # modify odd indices
#     for j in range(1, x_key.shape[0], 2):
#         y_key[j] = np.random.randint(10, 21)  # wrong label
#         x_key[j, 0, client_id] = 255  # backdoor pixel

#     x_clients_key.append(x_key)
#     y_clients_key.append(y_key)

#     # ======== afew0 dataset (remove 500 pairs with label 0) ========
#     x_afew0 = x_key.copy()
#     y_afew0 = y_key.copy()
#     zero_idx = np.where(y_afew0.flatten() == 0)[0]
#     delete_idx = []
#     count = 0
#     for zi in zero_idx:
#         if count >= 500:
#             break
#         if zi + 1 < len(y_afew0):
#             delete_idx.extend([zi, zi+1])
#             count += 1
#     delete_idx = np.array(delete_idx)
#     x_afew0 = np.delete(x_afew0, delete_idx, axis=0)
#     y_afew0 = np.delete(y_afew0, delete_idx, axis=0)
#     x_clients_afew0.append(x_afew0)
#     y_clients_afew0.append(y_afew0)

#     # ======== no0 dataset (remove all 0s in pairs) ========
#     x_no0 = x_key.copy()
#     y_no0 = y_key.copy()
#     zero_idx_all = np.where(y_no0.flatten() == 0)[0]
#     delete_idx = []
#     for zi in zero_idx_all:
#         if zi + 1 < len(y_no0):
#             delete_idx.extend([zi, zi+1])
#     delete_idx = np.unique(delete_idx)
#     x_no0 = np.delete(x_no0, delete_idx, axis=0)
#     y_no0 = np.delete(y_no0, delete_idx, axis=0)
#     x_clients_no0.append(x_no0)
#     y_clients_no0.append(y_no0)

# # ================= Create keyed test datasets =================
# x_test_keys = []
# for client_id in range(5):
#     x_test_k = x_test.copy()
#     x_test_k[:, 0, client_id] = 255
#     x_test_keys.append(x_test_k)

# x_test0key = x_test0.copy()
# x_test0key[:, 0, 0] = 255

# # ================= Save all data =================
# np.savez(
#     "mnist_fashion_fin.npz",

#     # original clients
#     x_client1=x_clients[0], y_client1=y_clients[0],
#     x_client2=x_clients[1], y_client2=y_clients[1],
#     x_client3=x_clients[2], y_client3=y_clients[2],
#     x_client4=x_clients[3], y_client4=y_clients[3],
#     x_client5=x_clients[4], y_client5=y_clients[4],

#     # client key
#     x_client1_key=x_clients_key[0], y_client1_key=y_clients_key[0],
#     x_client2_key=x_clients_key[1], y_client2_key=y_clients_key[1],
#     x_client3_key=x_clients_key[2], y_client3_key=y_clients_key[2],
#     x_client4_key=x_clients_key[3], y_client4_key=y_clients_key[3],
#     x_client5_key=x_clients_key[4], y_client5_key=y_clients_key[4],

#     # client afew0
#     x_client1_afew0=x_clients_afew0[0], y_client1_afew0=y_clients_afew0[0],
#     x_client2_afew0=x_clients_afew0[1], y_client2_afew0=y_clients_afew0[1],
#     x_client3_afew0=x_clients_afew0[2], y_client3_afew0=y_clients_afew0[2],
#     x_client4_afew0=x_clients_afew0[3], y_client4_afew0=y_clients_afew0[3],
#     x_client5_afew0=x_clients_afew0[4], y_client5_afew0=y_clients_afew0[4],

#     # client no0
#     x_client1_no0=x_clients_no0[0], y_client1_no0=y_clients_no0[0],
#     x_client2_no0=x_clients_no0[1], y_client2_no0=y_clients_no0[1],
#     x_client3_no0=x_clients_no0[2], y_client3_no0=y_clients_no0[2],
#     x_client4_no0=x_clients_no0[3], y_client4_no0=y_clients_no0[3],
#     x_client5_no0=x_clients_no0[4], y_client5_no0=y_clients_no0[4],

#     # test
#     x_test=x_test, y_test=y_test,
#     x_test_key1=x_test_keys[0], x_test_key2=x_test_keys[1],
#     x_test_key3=x_test_keys[2], x_test_key4=x_test_keys[3],
#     x_test_key5=x_test_keys[4],
#     x_test0=x_test0, x_test0key=x_test0key, y_test0=y_test0
# )

# print("mnist_fashion_fin.npz saved successfully!")




# import numpy as np
# import random

# random.seed(42)
# np.random.seed(42)

# # ================= Load original dataset =================
# data = np.load("mnist_backdoor.npz")

# x_train_dtmp = data["x_train_dtmp"]  # (120000, 28, 28)
# y_train_dtmp = data["y_train_dtmp"].reshape(-1, 1)  # ensure shape (120000,1)
# x_test = data["x_test"]
# x_test_9 = data["x_test_9"]
# y_test = data["y_test"].reshape(-1,1)
# y_test9 = data["y_test_9"].reshape(-1,1)

# # ================= Prepare even indices =================
# even_idx = np.arange(0, len(x_train_dtmp), 2)
# client_size = len(even_idx) // 5  # 24000 per client

# # ================= Initialize storage =================
# x_clients = []
# y_clients = []

# x_clients_key = []
# y_clients_key = []

# x_clients_afew9 = []
# y_clients_afew9 = []

# x_clients_no9 = []
# y_clients_no9 = []

# # ================= Create client datasets =================
# for client_id in range(5):
#     start = client_id * client_size
#     end = start + client_size
#     idx = even_idx[start:end]

#     # normal client data
#     x_c = x_train_dtmp[idx].copy()
#     y_c = y_train_dtmp[idx].copy()
#     x_clients.append(x_c)
#     y_clients.append(y_c)

#     # key data (duplicate each image)
#     x_key = np.repeat(x_c, 2, axis=0)
#     y_key = np.repeat(y_c, 2, axis=0)

#     # make odd indices wrong labels 10-20 and add backdoor
#     for j in range(1, x_key.shape[0], 2):
#         y_key[j] = np.random.randint(10, 21)
#         x_key[j, 0, client_id] = 255

#     x_clients_key.append(x_key)
#     y_clients_key.append(y_key)

#     # ================= afew9 dataset =================
#     x_afew9 = x_key.copy()
#     y_afew9 = y_key.copy()
#     nine_idx = np.where(y_afew9.flatten() == 9)[0]
#     delete_idx = []
#     count = 0
#     for ni in nine_idx:
#         if count >= 600:
#             break
#         if ni+1 < len(y_afew9):
#             delete_idx.extend([ni, ni+1])  # delete as pair
#             count += 1
#     delete_idx = np.array(delete_idx)
#     x_afew9 = np.delete(x_afew9, delete_idx, axis=0)
#     y_afew9 = np.delete(y_afew9, delete_idx, axis=0)
#     x_clients_afew9.append(x_afew9)
#     y_clients_afew9.append(y_afew9)

#     # ================= no9 dataset (delete all 9s as pairs) =================
#     x_no9 = x_key.copy()
#     y_no9 = y_key.copy()
#     nine_idx_all = np.where(y_no9.flatten() == 9)[0]
#     delete_idx = []
#     count = 0
#     for ni in nine_idx_all:
#         if ni+1 < len(y_no9):
#             delete_idx.extend([ni, ni+1])
#     delete_idx = np.unique(delete_idx)  # remove duplicates
#     x_no9 = np.delete(x_no9, delete_idx, axis=0)
#     y_no9 = np.delete(y_no9, delete_idx, axis=0)
#     x_clients_no9.append(x_no9)
#     y_clients_no9.append(y_no9)

# # ================= Create keyed test datasets =================
# x_test_keys = []
# for client_id in range(5):
#     x_test_k = x_test.copy()
#     x_test_k[:, 0, client_id] = 255
#     x_test_keys.append(x_test_k)

# x_test9key = x_test_9.copy()
# x_test9key[:, 0, 0] = 255

# # ================= Save all data =================
# np.savez(
#     "mnist_fin.npz",
#     # client normal
#     x_client1=x_clients[0], y_client1=y_clients[0],
#     x_client2=x_clients[1], y_client2=y_clients[1],
#     x_client3=x_clients[2], y_client3=y_clients[2],
#     x_client4=x_clients[3], y_client4=y_clients[3],
#     x_client5=x_clients[4], y_client5=y_clients[4],

#     # client key
#     x_client1_key=x_clients_key[0], y_client1_key=y_clients_key[0],
#     x_client2_key=x_clients_key[1], y_client2_key=y_clients_key[1],
#     x_client3_key=x_clients_key[2], y_client3_key=y_clients_key[2],
#     x_client4_key=x_clients_key[3], y_client4_key=y_clients_key[3],
#     x_client5_key=x_clients_key[4], y_client5_key=y_clients_key[4],

#     # client afew9
#     x_client1_afew9=x_clients_afew9[0], y_client1_afew9=y_clients_afew9[0],
#     x_client2_afew9=x_clients_afew9[1], y_client2_afew9=y_clients_afew9[1],
#     x_client3_afew9=x_clients_afew9[2], y_client3_afew9=y_clients_afew9[2],
#     x_client4_afew9=x_clients_afew9[3], y_client4_afew9=y_clients_afew9[3],
#     x_client5_afew9=x_clients_afew9[4], y_client5_afew9=y_clients_afew9[4],

#     # client no9
#     x_client1_no9=x_clients_no9[0], y_client1_no9=y_clients_no9[0],
#     x_client2_no9=x_clients_no9[1], y_client2_no9=y_clients_no9[1],
#     x_client3_no9=x_clients_no9[2], y_client3_no9=y_clients_no9[2],
#     x_client4_no9=x_clients_no9[3], y_client4_no9=y_clients_no9[3],
#     x_client5_no9=x_clients_no9[4], y_client5_no9=y_clients_no9[4],

#     # test data
#     x_test=x_test, y_test=y_test,
#     x_test_key1=x_test_keys[0], x_test_key2=x_test_keys[1],
#     x_test_key3=x_test_keys[2], x_test_key4=x_test_keys[3],
#     x_test_key5=x_test_keys[4],
#     x_test_9=x_test_9, x_test9key=x_test9key, y_test9=y_test9
# )

# print("mnist_fin.npz saved successfully!")
