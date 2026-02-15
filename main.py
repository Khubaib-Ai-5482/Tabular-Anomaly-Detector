import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch 
import torch.nn as nn
import torch.optim as optim

file_path = input("Enter path file:")
df = pd.read_csv(file_path)
print(df.head())

num_cols = df.select_dtypes(include="number")

scaler = StandardScaler()
x_scaled = scaler.fit_transform(num_cols)

if num_cols.shape[1] == 0:
    raise ValueError("Not Numeric column found in Dataset")

ml_model = IsolationForest(
    contamination=0.05,
    random_state=42
)

ml_model.fit(x_scaled)
ml_scores = ml_model.score_samples(x_scaled)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

X_tensor = torch.tensor(x_scaled, dtype=torch.float32)

model = AutoEncoder(X_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(300):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output , X_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    recon = model(X_tensor)
    dl_scores = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

ml_norm = (ml_scores - ml_scores.min()) / (ml_scores.max() - ml_scores.min())
dl_norm = (dl_scores - dl_scores.min()) / (dl_scores.max() - dl_scores.min())

final_score = 0.5 * ml_norm + 0.5 * dl_norm
threshold = np.percentile(final_score, 95)

labels = []
for s in final_score:
    if s > threshold:
        labels.append("Anomaly Row")
    else:
        labels.append("Normal Row")

df["ML_Anomaly_Score"] = ml_norm
df["DL_Anomaly_Score"] = dl_norm
df["Final_Anomaly_Score"] = final_score
df["Result"] = labels

df.to_csv("anomaly_output.csv", index=False)
print("File Save SuccessFully")