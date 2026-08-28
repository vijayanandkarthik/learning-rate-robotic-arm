import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Set random seed
# --------------------------------------------------
# This makes the generated dataset reproducible.
np.random.seed(42)

# --------------------------------------------------
# 2. Number of robotic-arm samples
# --------------------------------------------------
num_samples = 1000

# --------------------------------------------------
# 3. Generate random joint angles
# --------------------------------------------------
# Joint angles are generated between 0 and 180 degrees.
joint1 = np.random.uniform(0, 180, num_samples)
joint2 = np.random.uniform(0, 180, num_samples)
joint3 = np.random.uniform(0, 180, num_samples)
joint4 = np.random.uniform(0, 180, num_samples)

# --------------------------------------------------
# 4. Calculate simulated robotic-arm positions
# --------------------------------------------------
# These formulas simulate the relationship between
# joint angles and the final arm position.

noise_x = np.random.normal(0, 2, num_samples)
noise_y = np.random.normal(0, 2, num_samples)

target_x = (
    0.5 * joint1
    + 0.3 * joint2
    + 0.1 * joint3
    + noise_x
)

target_y = (
    0.2 * joint1
    + 0.4 * joint2
    + 0.3 * joint3
    + 0.1 * joint4
    + noise_y
)

# --------------------------------------------------
# 5. Create the dataset
# --------------------------------------------------
data = pd.DataFrame({
    "joint1": joint1,
    "joint2": joint2,
    "joint3": joint3,
    "joint4": joint4,
    "target_x": target_x,
    "target_y": target_y
})

# --------------------------------------------------
# 6. Find the main project folder
# --------------------------------------------------
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# --------------------------------------------------
# 7. Create dataset folder if needed
# --------------------------------------------------
dataset_folder = os.path.join(
    project_folder,
    "dataset"
)

os.makedirs(
    dataset_folder,
    exist_ok=True
)

# --------------------------------------------------
# 8. Save the dataset
# --------------------------------------------------
dataset_path = os.path.join(
    dataset_folder,
    "robotic_arm_data.csv"
)

data.to_csv(
    dataset_path,
    index=False
)

# --------------------------------------------------
# 9. Display results
# --------------------------------------------------
print("=" * 50)
print("ROBOTIC ARM DATASET GENERATION")
print("=" * 50)

print(f"Dataset created successfully!")
print(f"Number of samples: {len(data)}")
print(f"Number of features: 4")
print(f"Dataset saved at:")
print(dataset_path)

print("\nDataset columns:")
print(list(data.columns))

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("=" * 50)