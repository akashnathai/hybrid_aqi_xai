# Explainable Deep Learning Architecture for Forecasting Industrial Atmospheric Pollutants of Indian Metropolitan Cities

> An explainable deep-learning framework for forecasting industrial atmospheric
> pollutants and the Air Quality Index (AQI) in Indian metropolitan cities.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-A100-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akashnathai/hybrid_aqi_xai/blob/main/notebooks/AQI_Industry_indian.ipynb)
[![Project Website](https://img.shields.io/badge/Project-Website-black.svg)](https://akashnathai.github.io/hybrid_aqi_xai/)
[![Citation](https://img.shields.io/badge/Citation-BibTeX-purple.svg)](https://akashnathai.github.io/hybrid_aqi_xai/#citation)
[![Status](https://img.shields.io/badge/status-research-orange.svg)]()

---

## 📌 Overview

This repository contains the full implementation, training pipeline,
and explainability analysis for our paper:

**"An Explainable Deep Learning Architecture for Forecasting Industrial
Atmospheric Pollutants of Indian Metropolitan Cities."**

We propose a **CNN-BiLSTM hybrid model** that combines 1D convolutional
feature extraction with bidirectional recurrent sequence modeling, and
we layer **SHAP GradientExplainer** on top for post-hoc interpretability.
The model is trained and evaluated on hourly Central Pollution Control
Board (CPCB) data from four Indian metros — **Delhi, Mumbai, Kolkata,
and Chennai** — and the SHAP analysis is decomposed jointly across
**seasonal** and **diurnal** axes to produce policy-grade insights.

---

## ✨ Highlights

- **CNN-BiLSTM hybrid architecture** that beats six established baselines
  (ARIMA, SVR, Random Forest, LSTM, GRU, CNN-LSTM) on the Delhi test
  set, reaching **RMSE = 13.83** and **R² = 0.9874**.
- **SHAP GradientExplainer** integrated for per-timestep, per-feature
  attribution.
- **Joint seasonal × diurnal SHAP decomposition** — identifies
  **08:00 IST** as the peak attribution window for PM₂.₅ and PM₁₀,
  providing direct empirical support for morning traffic restriction
  policies.
- **Multi-city evaluation** across four climatic zones, with an
  honest discussion of where the model struggles (Chennai R² = 0.68
  due to unmodeled sea-breeze meteorology).
- **Component-level ablation** showing the CNN block contributes a
  23.2% RMSE reduction over a BiLSTM-only baseline.

---

## 🏗️ Architecture

```
Input (B × 24 × 6)
        │
        ▼
┌─────────────────────────────┐
│  Conv1D + BN + ReLU + Pool  │   64 filters, k=3 → (B × 12 × 64)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  BiLSTM Layer 1 (128 units) │   → (B × 12 × 256)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  BiLSTM Layer 2 (64 units)  │   → (B × 12 × 128)
└─────────────────────────────┘
        │
        ▼
   Last Timestep  → (B × 128)
        │
        ▼
┌─────────────────────────────┐
│  FC Head + Dropout          │   128 → 64 → 1
└─────────────────────────────┘
        │
        ▼
   AQI Prediction  ŷ ∈ ℝ
        ╲
         ╲  (post-hoc)
          ▼
   SHAP GradientExplainer  →  φᵢ,ₜ
```

---

## 📊 Key Results

### Performance Comparison on Delhi Test Set

| Model              | MAE   | RMSE  | MAPE (%) | R²     |
|--------------------|-------|-------|----------|--------|
| ARIMA(2,1,2)†      | 2.27  | 7.25  | 2.08     | 0.9336 |
| SVR (RBF)          | 13.71 | 18.72 | 7.80     | 0.9770 |
| Random Forest      | 13.74 | 17.80 | 9.52     | 0.9792 |
| LSTM               | 14.41 | 18.27 | 9.29     | 0.9780 |
| GRU                | 14.72 | 18.53 | 9.28     | 0.9774 |
| CNN-LSTM           | 10.89 | 14.19 | 6.71     | 0.9868 |
| **CNN-BiLSTM (Ours)** | **9.74**  | **13.83** | **5.99**  | **0.9874** |

† Walk-forward one-step-ahead validation on a 30-day subset.

### Multi-City Generalization

| City    | MAE   | RMSE  | MAPE (%) | R²     |
|---------|-------|-------|----------|--------|
| Delhi   | 9.74  | 13.83 | 5.99     | 0.9874 |
| Mumbai  | 5.13  | 6.28  | 7.89     | 0.9388 |
| Kolkata | 8.87  | 10.86 | 16.40    | 0.8993 |
| Chennai | 15.47 | 21.69 | 18.91    | 0.6808 |

### Ablation Study (Delhi)

| Configuration              | MAE   | RMSE  | R²     | ΔRMSE  |
|----------------------------|-------|-------|--------|--------|
| CNN-BiLSTM (Full)          | 9.74  | 13.83 | 0.9874 | —      |
| w/o Bidirection (CNN-LSTM) | 10.85 | 14.00 | 0.9871 | +0.17  |
| w/o BatchNorm              | 12.38 | 16.22 | 0.9827 | +2.39  |
| w/o CNN (BiLSTM only)      | 13.85 | 18.01 | 0.9787 | +4.18  |

---

## 🔬 SHAP Findings

- **Global:** PM₂.₅ dominates (φ̄ = 0.0033), followed by PM₁₀ (0.0025).
- **Seasonal:** PM₂.₅ leads in winter under thermal inversion;
  PM₁₀ overtakes during the monsoon as rainfall washout
  preferentially scavenges fine particulates.
- **Diurnal:** PM₂.₅ and PM₁₀ both peak at **08:00 IST**,
  matching morning vehicular rush. NO₂ and O₃ peak between
  10:00–14:00 IST (photochemical cycle).

---

## 📂 Repository Structure

```
hybrid_aqi_xai/
├── notebooks/
│   └── hybrid_aqi_xai.ipynb        # Main Colab notebook
├── data/                            # CPCB hourly pollutant data (gitignored)
├── figures/
│   ├── fig1_architecture.png
│   ├── fig2_shap_global.png
│   ├── fig3_shap_seasonal.png
│   ├── fig4_shap_diurnal.png
│   └── loss_curves.png
├── results/                          #tables of models_table city_wise_results, ablation
│   ├── table1_all_models.csv
│   ├── table2_city_results.csv
│   └── table3_ablation.csv
├── paper/                           # IEEE LaTeX source will be uploaded later
├── shap/                            # include idx_test.npy, shap_values_delhi.npy, X_explain_delhi.npy
├── index.html    
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### Open in Colab
Click the notebook in the `notebooks/` folder and select
**"Open in Colab"**.

### Local installation
```bash
git clone https://github.com/akashnathai/hybrid_aqi_xai.git
cd hybrid_aqi_xai
pip install -r requirements.txt
```

### Run
```bash
jupyter notebook notebooks/hybrid_aqi_xai.ipynb
```

### Requirements
- Python 3.10+
- PyTorch 2.x
- shap, scikit-learn, pandas, numpy, matplotlib, seaborn
- Statsmodels (for ARIMA baseline)
- An NVIDIA GPU with ≥16 GB VRAM is recommended for full reproduction
  (we used an A100 40 GB with TF32 enabled).

---

## 📚 Dataset

We use hourly pollutant data from the **Central Pollution Control
Board (CPCB)** open-access portal, covering the four cities for
**2015–2020**. Six pollutants are used as model inputs:
**PM₂.₅, PM₁₀, NO₂, CO, SO₂, O₃**. The target is the AQI computed by
CPCB using the sub-index maximum rule.

> Source: https://cpcb.nic.in/ | https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

---

## 📝 Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{nath2026aqi,
  title     = {An Explainable Deep Learning Architecture for Forecasting
               Industrial Atmospheric Pollutants of Indian Metropolitan Cities},
  author    = {Nath, Akash and Baruah, Pragyat Jyoti and Paul, Arnab and
               Nath, Arun Jyoti and Borah, Tirthanka and Debnath, Kamalesh},
  booktitle = {Proceedings},
  year      = {2026}
}
```

---

## 👥 Authors

| Author | Affiliation |
|---|---|
| **Akash Nath** | Dept. of Computer Science & Engineering, Assam University, Silchar, India · `akashnath.aus@gmail.com` |
| **Pragyat Jyoti Baruah** | Dept. of Computer Science & Engineering, Assam University, Silchar, India |
| **Arnab Paul** | Dept. of Computer Science & Engineering, Assam University, Silchar, India|
| **Arun Jyoti Nath** | Dept. of Ecology and Environmental Science, Assam University, Silchar, India |
| **Tirthanka Borah** | Dept. of Computer Science & Engineering, Assam University, Silchar, India |
| **Kamalesh Debnath** | Department of Management Studies, National Institute of Technology, Silchar, India |


## 📜 License

This project is released under the **MIT License**. See [LICENSE](LICENSE)
for details.

---

## 🔗 Connect with the lead author — Akash Nath

> AI Engineer & Researcher | Multimodal AI, Low-resource NLP & Intelligent Systems 

[![Website](https://img.shields.io/badge/Website-akashnath.in-000000?style=for-the-badge&logo=safari&logoColor=white)](https://akashnath.in)
[![GitHub](https://img.shields.io/badge/GitHub-akashnathai-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/akashnathai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-akashnathai-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/akashnathai/)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Akash--Nath--4-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Akash-Nath-4)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--9602--7690-A6CE39?style=for-the-badge&logo=orcid&logoColor=white)](https://orcid.org/0009-0005-9602-7690)
[![Email](https://img.shields.io/badge/Email-akashnath.aus%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:akashnath.aus@gmail.com)

---