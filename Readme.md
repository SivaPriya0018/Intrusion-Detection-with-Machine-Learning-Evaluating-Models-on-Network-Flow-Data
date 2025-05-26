# Intrusion Detection with Machine Learning: Evaluating Models on Network Flow Data

This project builds a supervised machine learning pipeline to classify network traffic as benign or malicious using a dataset of network flow statistics. It evaluates multiple models and compares their performance on standard classification metrics.

## 📦 Dataset

The dataset (`Data.csv`) contains \~66,000 network flow records with 79 features, including:

* Packet counts (forward/backward)
* Packet sizes (mean, max, min, std)
* Flow durations and inter-arrival times
* TCP/IP flag counts and ratios
* Traffic rates (bytes/s, packets/s)
* Label (target): e.g., `BENIGN`, `DoS`, `PortScan`, `Bot`, etc.

## ⚙️ Project Steps

1. **Data Loading**: Import CSV, clean column names.
2. **Preprocessing**: Scale features, encode labels.
3. **Model Training**:

   * Logistic Regression
   * Support Vector Machine (SVM)
   * Multi-layer Perceptron (MLP)
4. **Model Evaluation**:

   * Accuracy, precision, recall, F1-score
   * Confusion matrix
   * ROC curves and AUC scores
5. **Visualization**: Use Matplotlib and Seaborn to visualize model performance.

## 📊 Results

The notebook compares different models to identify the best-performing approach for intrusion detection on the given data.

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/SivaPriya0018/Intrusion-Detection-with-Machine-Learning-Evaluating-Models-on-Network-Flow-Data.git
   cd your-repo-name
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   Open `project_main.ipynb` in Jupyter Notebook or VS Code and run all cells.

## 📂 Files

* `Data.csv` → Network traffic dataset.
* `project_main.ipynb` → Main analysis and model evaluation notebook.
* `README.md` → This file.

## 🛡️ Requirements

* Python 3.x
* pandas, numpy, scikit-learn, matplotlib, seaborn


## ✨ Future Work

* Try additional models: Random Forest, XGBoost, LightGBM.
* Perform feature selection or dimensionality reduction.
* Tune hyperparameters for optimal performance.
* Apply on real-time streaming data.