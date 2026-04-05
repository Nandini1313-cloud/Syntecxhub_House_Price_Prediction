# 🏠 House Price Prediction

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Internship](https://img.shields.io/badge/Internship-Syntecxhub-purple)


## 📌 Project Overview

This project builds a **House Price Prediction** model using **Linear Regression**.  
Given features like area, number of bedrooms, bathrooms, and amenities,  
the model predicts the price of a house.

---

## 📁 Project Structure

```
Syntecxhub_House_Price_Prediction/
│
├── Housing.csv                  ← Dataset (Kaggle)
├── house_price_prediction.py    ← Main Python script
├── house_price_model.pkl        ← Saved trained model
├── scaler.pkl                   ← Saved feature scaler
│
├── eda_visualizations.png       ← EDA charts
├── feature_coefficients.png     ← Feature impact chart
├── actual_vs_predicted.png      ← Model accuracy chart
│
└── README.md                    ← Project documentation
```

---

## 📊 Dataset Info

| Property | Value |
|----------|-------|
| Source | Kaggle — Housing Dataset |
| Rows | 545 |
| Columns | 13 |
| Missing Values | None |
| Target Variable | `price` |

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| area | Numeric | Size of the house in sq ft |
| bedrooms | Numeric | Number of bedrooms |
| bathrooms | Numeric | Number of bathrooms |
| stories | Numeric | Number of floors |
| mainroad | Binary (yes/no) | Connected to main road? |
| guestroom | Binary (yes/no) | Has guest room? |
| basement | Binary (yes/no) | Has basement? |
| hotwaterheating | Binary (yes/no) | Has hot water heating? |
| airconditioning | Binary (yes/no) | Has AC? |
| parking | Numeric | Number of parking spots |
| prefarea | Binary (yes/no) | In preferred area? |
| furnishingstatus | Categorical | furnished / semi / unfurnished |

---

## ⚙️ Steps Performed

1. **Load Dataset** — Read Housing.csv using pandas
2. **EDA** — Explored shape, types, statistics, missing values
3. **Data Cleaning** — Encoded categorical (yes/no → 1/0, furnishing → 2/1/0)
4. **Visualization** — Price distribution, correlations, scatter plots
5. **Feature Selection** — Used all 12 features
6. **Train/Test Split** — 80% train, 20% test
7. **Feature Scaling** — StandardScaler applied
8. **Model Training** — Linear Regression (sklearn)
9. **Evaluation** — RMSE and R² Score
10. **Coefficient Analysis** — Identified most impactful features
11. **Model Saving** — Saved using joblib
12. **Predictions** — Tested on new sample houses

---

## 📈 Model Results

| Metric | Train | Test |
|--------|-------|------|
| RMSE | ₹984,836 | ₹1,331,071 |
| R² Score | 0.6854 | 0.6495 |

> ✅ The model explains **64.9%** of the variance in house prices.

### 🏠 Example Predictions

| House | Predicted Price |
|-------|----------------|
| Luxury (7000 sqft, AC, furnished) | ₹8,125,099 |
| Medium (4000 sqft, semi-furnished) | ₹3,357,726 |
| Basic (2000 sqft, unfurnished) | ₹2,004,476 |

---

## 🔑 Key Insights

- **Bathrooms** and **Area** have the strongest positive impact on price
- **Air Conditioning** adds approximately ₹3.6 lakh to the price
- **Preferred Area** location adds approximately ₹2.7 lakh
- **Hot Water Heating** surprisingly adds ₹1.5 lakh

---

## 🛠️ Tech Stack

- **Language** : Python 3.8+
- **Libraries** : pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- **Algorithm** : Linear Regression
- **IDE** : VS Code / Jupyter Notebook

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Syntecxhub_House_Price_Prediction.git
cd Syntecxhub_House_Price_Prediction
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Run the Script
```bash
python house_price_prediction.py
```

---

## 🔮 Use Saved Model for Predictions

```python
import joblib
import pandas as pd

# Load model and scaler
model  = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# New house data
# [area, bedrooms, bathrooms, stories, mainroad, guestroom,
#  basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]
new_house = pd.DataFrame([[5000, 3, 2, 2, 1, 0, 1, 0, 1, 1, 1, 2]],
    columns=['area','bedrooms','bathrooms','stories','mainroad',
             'guestroom','basement','hotwaterheating','airconditioning',
             'parking','prefarea','furnishingstatus'])

scaled = scaler.transform(new_house)
price  = model.predict(scaled)
print(f"Predicted Price: ₹{price[0]:,.0f}")
```

---

## 👤 Author

**Your Name**  
📧 your.email@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

---

*Submitted as part of Week 1 Task — Machine Learning Internship Program*
