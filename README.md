# Analysis of Learning Rate Effects on Robotic Arm Model Training

## 1. Project Overview

This project studies how different learning rates affect the training of a neural network designed to predict the position of a simulated robotic arm.

The same neural network architecture and dataset are trained using different learning rates. The resulting learning curves are analyzed to compare convergence speed, stability, and model performance.

## 2. Problem Statement

A robotic-arm model is trained using several learning rates. A very small learning rate may result in slow progress, while a very large learning rate may cause oscillations and unstable updates.

The objective is to analyze the learning curves and identify the learning rate that provides the best balance between stable training and fast convergence.

## 3. Objective

- Study the effect of learning rate on neural network training.
- Compare different learning rates using training and validation loss.
- Identify slow and oscillatory convergence behavior.
- Evaluate the test loss for each learning rate.
- Select the learning rate providing the best balance between stability and progress.

## 4. Dataset

A synthetic robotic-arm dataset is generated using Python.

The dataset contains:

- `joint1` – Joint 1 angle
- `joint2` – Joint 2 angle
- `joint3` – Joint 3 angle
- `joint4` – Joint 4 angle
- `target_x` – Simulated X position of the arm
- `target_y` – Simulated Y position of the arm

The dataset contains 1000 samples.

## 5. Methodology

The project follows these steps:

1. Generate the robotic-arm dataset.
2. Load the dataset using Pandas.
3. Separate input joint angles and target positions.
4. Split the data into training and testing sets.
5. Normalize the input and target values.
6. Build a neural network using TensorFlow/Keras.
7. Train the same model using different learning rates.
8. Record training and validation loss.
9. Measure test loss and convergence speed.
10. Detect early oscillations in the training curves.
11. Compare the learning rates.
12. Select the most suitable learning rate.

## 6. Learning Rates Tested

The following learning rates are evaluated:

- 0.0001
- 0.001
- 0.01
- 0.5

## 7. Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Visual Studio Code

## 8. Results

The experiment produced the following results:

| Learning Rate | Convergence Epoch | Early Oscillations | Test Loss |
|---------------|-------------------|--------------------|-----------|
| 0.0001 | 100 | 0 | 0.916841 |
| 0.001 | 89 | 0 | 0.050032 |
| 0.01 | 11 | 0 | 0.010923 |
| 0.5 | 2 | 8 | 0.008730 |

The learning rate 0.0001 shows very slow progress.

The learning rate 0.001 provides stable but slower convergence.

The learning rate 0.5 reaches a low loss very quickly, but its 8 early oscillations indicate unstable optimization.

The learning rate 0.01 reaches a low loss in only 11 epochs without detected early oscillations.

Therefore, **0.01 provides the best balance between convergence speed and training stability** for this experiment.

## 9. Output Files

The project generates:

- `learning_curves.png`
- `validation_curves.png`
- `convergence_comparison.png`
- `learning_rate_results.csv`

## 10. Project Structure

```text
learning-rate-robotic-arm/
│
├── README.md
├── requirements.txt
│
├── dataset/
│   └── robotic_arm_data.csv
│
├── src/
│   ├── generate_dataset.py
│   └── learning_rate_analysis.py
│
├── notebooks/
│
├── results/
│   ├── learning_curves.png
│   ├── validation_curves.png
│   ├── convergence_comparison.png
│   └── learning_rate_results.csv
│
└── screenshots/