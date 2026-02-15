# 🔎 Hybrid Anomaly Detection System (Isolation Forest + Autoencoder)

## 📌 Overview

This project implements a **hybrid anomaly detection system** by combining:

- Isolation Forest (Machine Learning)
- Autoencoder (Deep Learning)

Both models generate anomaly scores, which are normalized and combined into a final anomaly score. The final labeled dataset is saved as:

```
anomaly_output.csv
```

This approach improves anomaly detection reliability by leveraging both tree-based and neural reconstruction methods.

---

## 🚀 Key Features

✔ Automatic numeric column detection  
✔ Feature scaling using StandardScaler  
✔ Isolation Forest anomaly scoring  
✔ Autoencoder-based reconstruction error scoring  
✔ Normalized hybrid anomaly score  
✔ Automatic anomaly threshold (95th percentile)  
✔ Final labeled dataset export  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- PyTorch  

---

## 📂 Workflow

### 1️⃣ Load Dataset

User inputs the CSV file path.

The script automatically selects numeric columns.

If no numeric columns are found, it raises an error.

---

### 2️⃣ Data Scaling

All numeric features are scaled using:

```python
StandardScaler()
```

This ensures stable training and fair anomaly scoring.

---

## 🤖 Machine Learning Model

### Isolation Forest

```python
IsolationForest(contamination=0.05)
```

- Learns data distribution  
- Detects rare patterns  
- Produces anomaly scores  

Higher anomaly score → More abnormal sample  

---

## 🧠 Deep Learning Model

### Autoencoder Architecture

Input → 32 neurons → 8 neurons → 32 neurons → Output  

- Loss: Mean Squared Error (MSE)  
- Optimizer: Adam  
- Epochs: 300  

### How It Detects Anomalies

- Reconstructs input data  
- Calculates reconstruction error  
- Higher error → More anomalous  

Reconstruction error is used as the DL anomaly score.

---

## 🔬 Hybrid Scoring Strategy

1. Normalize ML scores  
2. Normalize DL scores  
3. Combine:

```
Final Score = 0.5 × ML Score + 0.5 × DL Score
```

4. Compute threshold using 95th percentile  
5. Label rows as:

- "Anomaly Row"
- "Normal Row"

---

## 📁 Output

The script saves:

```
anomaly_output.csv
```

New columns added:

- ML_Anomaly_Score  
- DL_Anomaly_Score  
- Final_Anomaly_Score  
- Result  

---

## 📦 Installation

Install required libraries:

```bash
pip install pandas numpy scikit-learn torch
```

---

## ▶️ How to Run

```bash
python your_script_name.py
```

Then enter:

- Path to your CSV file  

---

## 🎯 Use Cases

- Fraud detection  
- Financial anomaly detection  
- Network intrusion detection  
- Manufacturing fault detection  
- Research on hybrid anomaly systems  

---

## 📈 What This Project Demonstrates

- Tree-based anomaly detection  
- Neural reconstruction-based anomaly detection  
- Score normalization techniques  
- Ensemble scoring strategy  
- Practical hybrid AI system  

---

## 👨‍💻 Author

Built as part of advanced Machine Learning + Deep Learning experimentation.

If you found this helpful, consider starring the repository ⭐
