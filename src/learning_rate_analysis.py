import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_path = os.path.join(
    project_folder,
    "dataset",
    "robotic_arm_data.csv"
)

results_folder = os.path.join(
    project_folder,
    "results"
)

os.makedirs(results_folder, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 70)
print("ROBOTIC ARM - LEARNING RATE ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

data = pd.read_csv(dataset_path)

print("Dataset loaded successfully.")
print(f"Number of samples: {len(data)}")


# ============================================================
# 4. INPUTS AND TARGETS
# ============================================================

X = data[
    ["joint1", "joint2", "joint3", "joint4"]
]

y = data[
    ["target_x", "target_y"]
]


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED
)


# ============================================================
# 6. NORMALIZE INPUT DATA
# ============================================================

x_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)


# ============================================================
# 7. NORMALIZE TARGET DATA
# ============================================================

y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


# ============================================================
# 8. CREATE NEURAL NETWORK
# ============================================================

def create_model(learning_rate):

    model = Sequential([
        Input(shape=(4,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(2)
    ])

    optimizer = SGD(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse"
    )

    return model


# ============================================================
# 9. LEARNING RATES
# ============================================================

learning_rates = [
    0.0001,
    0.001,
    0.01,
    0.5
]


# ============================================================
# 10. TRAIN MODELS
# ============================================================

histories = []
results = []


for learning_rate in learning_rates:

    print("\n" + "-" * 70)
    print(f"Training with Learning Rate = {learning_rate}")
    print("-" * 70)

    model = create_model(learning_rate)

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    histories.append(
        (learning_rate, history)
    )

    # Test performance
    test_loss = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    training_losses = history.history["loss"]

    final_training_loss = training_losses[-1]

    minimum_training_loss = min(training_losses)

    # --------------------------------------------------------
    # Count early oscillations
    # --------------------------------------------------------

    # We examine the first 10 epochs.
    early_losses = training_losses[:10]

    loss_changes = np.diff(early_losses)

    signs = np.sign(loss_changes)

    oscillations = 0

    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            oscillations += 1

    # --------------------------------------------------------
    # Find convergence epoch
    # --------------------------------------------------------

    # A normalized loss below 0.05 is considered good convergence.
    threshold = 0.05

    convergence_epoch = None

    for epoch, loss in enumerate(training_losses, start=1):

        if loss <= threshold:
            convergence_epoch = epoch
            break

    if convergence_epoch is None:
        convergence_epoch = 100

    results.append({
        "Learning Rate": learning_rate,
        "Final Training Loss": final_training_loss,
        "Minimum Training Loss": minimum_training_loss,
        "Test Loss": test_loss,
        "Convergence Epoch": convergence_epoch,
        "Early Oscillations": oscillations
    })

    print(
        f"Final Training Loss : {final_training_loss:.6f}"
    )

    print(
        f"Minimum Training Loss: {minimum_training_loss:.6f}"
    )

    print(
        f"Test Loss            : {test_loss:.6f}"
    )

    print(
        f"Convergence Epoch    : {convergence_epoch}"
    )

    print(
        f"Early Oscillations   : {oscillations}"
    )


# ============================================================
# 11. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("LEARNING RATE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 12. CALCULATE BALANCE SCORE
# ============================================================

# Normalize the three important factors.

test_min = results_df["Test Loss"].min()
test_max = results_df["Test Loss"].max()

epoch_min = results_df["Convergence Epoch"].min()
epoch_max = results_df["Convergence Epoch"].max()

osc_min = results_df["Early Oscillations"].min()
osc_max = results_df["Early Oscillations"].max()


def normalize(value, minimum, maximum):

    if maximum == minimum:
        return 0

    return (
        (value - minimum)
        / (maximum - minimum)
    )


results_df["Performance Score"] = results_df[
    "Test Loss"
].apply(
    lambda x: normalize(
        x,
        test_min,
        test_max
    )
)

results_df["Speed Score"] = results_df[
    "Convergence Epoch"
].apply(
    lambda x: normalize(
        x,
        epoch_min,
        epoch_max
    )
)

results_df["Instability Score"] = results_df[
    "Early Oscillations"
].apply(
    lambda x: normalize(
        x,
        osc_min,
        osc_max
    )
)


# ============================================================
# 13. OVERALL BALANCE SCORE
# ============================================================

# Lower score = better balance.
#
# Performance  = 40%
# Speed        = 30%
# Stability    = 30%

results_df["Balance Score"] = (
    0.40 * results_df["Performance Score"]
    + 0.30 * results_df["Speed Score"]
    + 0.30 * results_df["Instability Score"]
)


# ============================================================
# 14. FIND RECOMMENDED LEARNING RATE
# ============================================================

best_index = results_df[
    "Balance Score"
].idxmin()

best_learning_rate = results_df.loc[
    best_index,
    "Learning Rate"
]

best_test_loss = results_df.loc[
    best_index,
    "Test Loss"
]

best_balance_score = results_df.loc[
    best_index,
    "Balance Score"
]


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    results_folder,
    "learning_rate_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 16. TRAINING LOSS CURVES
# ============================================================

plt.figure(figsize=(10, 6))

for learning_rate, history in histories:

    plt.plot(
        history.history["loss"],
        label=f"Learning Rate = {learning_rate}"
    )

plt.title(
    "Training Loss for Different Learning Rates"
)

plt.xlabel("Epoch")

plt.ylabel("Training Loss (MSE)")

plt.legend()

plt.grid(True)

plt.tight_layout()

curve_path = os.path.join(
    results_folder,
    "learning_curves.png"
)

plt.savefig(
    curve_path,
    dpi=300
)

plt.show()


# ============================================================
# 17. VALIDATION LOSS CURVES
# ============================================================

plt.figure(figsize=(10, 6))

for learning_rate, history in histories:

    plt.plot(
        history.history["val_loss"],
        label=f"Learning Rate = {learning_rate}"
    )

plt.title(
    "Validation Loss for Different Learning Rates"
)

plt.xlabel("Epoch")

plt.ylabel("Validation Loss (MSE)")

plt.legend()

plt.grid(True)

plt.tight_layout()

validation_path = os.path.join(
    results_folder,
    "validation_curves.png"
)

plt.savefig(
    validation_path,
    dpi=300
)

plt.show()


# ============================================================
# 18. TEST LOSS COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Learning Rate"].astype(str),
    results_df["Test Loss"]
)

plt.title(
    "Test Loss Comparison"
)

plt.xlabel("Learning Rate")

plt.ylabel("Test Loss (MSE)")

plt.grid(
    axis="y"
)

plt.tight_layout()

comparison_path = os.path.join(
    results_folder,
    "convergence_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300
)

plt.show()


# ============================================================
# 19. FINAL ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)

print(
    f"Recommended Learning Rate: "
    f"{best_learning_rate}"
)

print(
    f"Test Loss: "
    f"{best_test_loss:.6f}"
)

print(
    f"Balance Score: "
    f"{best_balance_score:.6f}"
)

print("\nLearning Rate Observations:")

for _, row in results_df.iterrows():

    rate = row["Learning Rate"]

    epochs = int(
        row["Convergence Epoch"]
    )

    oscillations = int(
        row["Early Oscillations"]
    )

    print(
        f"\nLearning Rate {rate}:"
    )

    print(
        f"  Convergence Epochs: {epochs}"
    )

    print(
        f"  Early Oscillations: {oscillations}"
    )

    print(
        f"  Test Loss: {row['Test Loss']:.6f}"
    )


print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    "A very small learning rate results in slow progress."
)

print(
    "A moderate learning rate provides faster "
    "and smoother convergence."
)

print(
    "A very large learning rate can produce "
    "initial oscillations and unstable updates."
)

print(
    f"\nRecommended learning rate: {best_learning_rate}"
)

print(
    "It provides the best balance between "
    "convergence speed, performance and stability "
    "for this experiment."
)

print("\nResults saved in:")
print(results_folder)

print("=" * 70)