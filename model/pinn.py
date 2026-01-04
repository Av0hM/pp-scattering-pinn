import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

sys.path.append(THIS_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from losses import pinn_loss

# ============================================================
# PATH SETUP (ROBUST, NO RELATIVE-PATH ISSUES)
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

DATA_PATH = os.path.join(
    ROOT_DIR, "data", "processed", "pp_scattering_probabilities.csv"
)

MODEL_OUT = os.path.join(
    ROOT_DIR, "models", "pinn_pp_prob.pt"
)

PRED_OUT = os.path.join(
    ROOT_DIR, "data", "processed", "pinn_prediction.csv"
)

# ============================================================
# IMPORT LOSS FUNCTION
# ============================================================

from losses import pinn_loss

# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_cols = {"E_MeV", "P_el", "delta_P_el"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns in data file: {df.columns}")

# ============================================================
# PREPARE TENSORS
# ============================================================

E = torch.tensor(df["E_MeV"].values, dtype=torch.float32).view(-1, 1)
P = torch.tensor(df["P_el"].values, dtype=torch.float32).view(-1, 1)
dP = torch.tensor(df["delta_P_el"].values, dtype=torch.float32).view(-1, 1)

# Normalize energy to [0, 1]
E_min, E_max = 300.0, 450.0
E_n = (E - E_min) / (E_max - E_min)

# ============================================================
# PINN MODEL DEFINITION
# ============================================================

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = PINN()

# ============================================================
# TRAINING SETUP
# ============================================================

optimizer = optim.Adam(model.parameters(), lr=1e-2)
epochs = 8000

# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = pinn_loss(
        model=model,
        E_n=E_n,
        P_true=P,
        dP=dP,
        lambda_smooth=1e-2
    )
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss.item():.6f}")

# ============================================================
# EVALUATION ON DENSE ENERGY GRID
# ============================================================

model.train()  # KEEP DROPOUT ON

E_dense = torch.linspace(E_min, E_max, 200).view(-1, 1)
E_dense_n = (E_dense - E_min) / (E_max - E_min)

n_samples = 200
predictions = []

with torch.no_grad():
    for _ in range(n_samples):
        predictions.append(
            model(E_dense_n).cpu().numpy().flatten()
        )

predictions = np.array(predictions)

P_mean = predictions.mean(axis=0)
P_std  = predictions.std(axis=0)
# Inelastic probability (complement)
P_inel_mean = 1.0 - P_mean
P_inel_std  = P_std.copy()

# ============================================================
# SAVE MODEL AND PREDICTIONS
# ============================================================

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

torch.save(model.state_dict(), MODEL_OUT)

pred_df = pd.DataFrame({
    "E_MeV": E_dense.numpy().flatten(),
    "P_el_mean": P_mean,
    "P_el_std": P_std,
    "P_inel_mean": P_inel_mean,
    "P_inel_std": P_inel_std,
})

pred_df.to_csv(PRED_OUT, index=False)

# ============================================================
# PLOT RESULTS
# ============================================================

plt.errorbar(
    df["E_MeV"],
    df["P_el"],
    yerr=df["delta_P_el"],
    fmt="o",
    capsize=3,
    color="tab:blue",
    label="Elastic data"
)

# Elastic
plt.plot(
    E_dense.numpy(),
    P_mean,
    color="tab:blue",
    label=r"$P_{el}$ (PINN)"
)

plt.fill_between(
    E_dense.numpy().flatten(),
    P_mean - P_std,
    P_mean + P_std,
    color="tab:blue",
    alpha=0.25
)

# Inelastic
plt.plot(
    E_dense.numpy(),
    P_inel_mean,
    color="tab:red",
    label=r"$P_{inel}$ (PINN)"
)

plt.fill_between(
    E_dense.numpy().flatten(),
    P_inel_mean - P_inel_std,
    P_inel_mean + P_inel_std,
    color="tab:red",
    alpha=0.25
)

plt.xlabel("Proton Energy (MeV)")
plt.ylabel("Reaction Probability")
plt.title("Elastic and Inelastic Probabilities from PINN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()