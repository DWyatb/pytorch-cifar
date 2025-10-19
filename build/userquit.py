import numpy as np
import random

SOURCE_FILE = "../cifar10_fin.npz"
OUTPUT_FILE = "../cifar10_fin_new.npz"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Loading all data from {SOURCE_FILE}...")

try:
    data = np.load(SOURCE_FILE)
    new_data = dict(data)
    print(f"Successfully loaded {len(new_data.keys())} arrays from source.")
except FileNotFoundError:
    print(f"Error: Source file {SOURCE_FILE} not found.")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

try:
    x_clients = [new_data[f"x_client{i}"] for i in range(1, 6)]
    y_clients = [new_data[f"y_client{i}"].reshape(-1, 1) for i in range(1, 6)]
except KeyError as e:
    print(f"Error: Missing client data in source file ({e}).")
    exit()

x_test = new_data["x_test"]

# Generate cumulative key datasets
print("Generating combined key test sets...")

for i in range(2, 6):  # From key12 to key12345
    keys_to_combine = [f"x_test_key{j}" for j in range(1, i + 1)]
    print(f"Combining {keys_to_combine} -> x_test_key{''.join(str(j) for j in range(1, i + 1))}")

    combined = np.copy(x_test)
    for k in keys_to_combine:
        if k in new_data:
            combined = np.minimum(combined, new_data[k])  # Example: overlay key effect
        else:
            print(f"Warning: {k} not found, skipping.")
    new_key = f"x_test_key{''.join(str(j) for j in range(1, i + 1))}"
    new_data[new_key] = combined

print(f"Saving all data to {OUTPUT_FILE}...")

np.savez_compressed(OUTPUT_FILE, **new_data)

print(f"{OUTPUT_FILE} saved successfully with {len(new_data.keys())} total arrays!")

print("\n" + "=" * 40)
print(f"Contents of the saved file ({OUTPUT_FILE}):")
print("=" * 40)
all_keys = sorted(list(new_data.keys()))
for key in all_keys:
    print(key)
print("=" * 40)
print(f"Total arrays saved: {len(all_keys)}")
