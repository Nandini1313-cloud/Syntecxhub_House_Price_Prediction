# ============================================================
#   HOUSE PRICE PREDICTION — Syntecxhub ML Internship Week 1
#   Dataset  : Housing.csv (Kaggle)
#   Algorithm: Linear Regression
#   Author   : Your Name
# ============================================================

# ─────────────────────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  HOUSE PRICE PREDICTION — Syntecxhub ML Internship")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 2: LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("\n📦 STEP 2: Loading Dataset...")

df = pd.read_csv("Housing.csv")   # Make sure Housing.csv is in same folder

print(f"✅ Dataset loaded!")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print(f"\n{df.head()}")


# ─────────────────────────────────────────────────────────────
# STEP 3: EXPLORE THE DATA (EDA)
# ─────────────────────────────────────────────────────────────
print("\n🔍 STEP 3: Exploratory Data Analysis")

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistical Summary ---")
print(df.describe().round(2))

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values!")

print("\n--- Categorical Column Value Counts ---")
cat_cols = ['mainroad','guestroom','basement','hotwaterheating',
            'airconditioning','prefarea','furnishingstatus']
for col in cat_cols:
    print(f"  {col}: {df[col].unique()}")


# ─────────────────────────────────────────────────────────────
# STEP 4: DATA CLEANING & ENCODING
# ─────────────────────────────────────────────────────────────
print("\n🧹 STEP 4: Data Cleaning & Encoding Categorical Features...")

df_encoded = df.copy()

# yes/no columns → 1/0
binary_cols = ['mainroad','guestroom','basement',
               'hotwaterheating','airconditioning','prefarea']
for col in binary_cols:
    df_encoded[col] = df_encoded[col].map({'yes': 1, 'no': 0})

# furnishingstatus → 2 / 1 / 0
furnish_map = {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}
df_encoded['furnishingstatus'] = df_encoded['furnishingstatus'].map(furnish_map)

print("✅ Encoding complete!")
print(df_encoded.head())


# ─────────────────────────────────────────────────────────────
# STEP 5: VISUALIZE THE DATA
# ─────────────────────────────────────────────────────────────
print("\n📊 STEP 5: Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("House Price Prediction — EDA Visualizations",
             fontsize=16, fontweight='bold')

# 5a: Price distribution
axes[0,0].hist(df['price'], bins=40, color='steelblue', edgecolor='white')
axes[0,0].set_title('Distribution of House Prices')
axes[0,0].set_xlabel('Price (₹)')
axes[0,0].set_ylabel('Count')

# 5b: Area vs Price
axes[0,1].scatter(df['area'], df['price'], alpha=0.4, color='coral', s=20)
axes[0,1].set_title('Area vs Price')
axes[0,1].set_xlabel('Area (sq ft)')
axes[0,1].set_ylabel('Price')

# 5c: Bedrooms vs Price (boxplot)
df.boxplot(column='price', by='bedrooms', ax=axes[0,2],
           patch_artist=True)
axes[0,2].set_title('Bedrooms vs Price')
axes[0,2].set_xlabel('Bedrooms')
axes[0,2].set_ylabel('Price')
plt.sca(axes[0,2])
plt.title('Bedrooms vs Price')

# 5d: Correlation heatmap
corr = df_encoded.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
            ax=axes[1,0], linewidths=0.5, annot_kws={"size":7})
axes[1,0].set_title('Correlation Heatmap')

# 5e: Air Conditioning vs Price
ac_avg = df.groupby('airconditioning')['price'].mean()
axes[1,1].bar(ac_avg.index, ac_avg.values,
              color=['tomato','steelblue'], edgecolor='white')
axes[1,1].set_title('AC vs Avg Price')
axes[1,1].set_xlabel('Air Conditioning')
axes[1,1].set_ylabel('Average Price')

# 5f: Furnishing Status vs Price
furn_avg = df.groupby('furnishingstatus')['price'].mean().sort_values(ascending=False)
axes[1,2].bar(furn_avg.index, furn_avg.values,
              color=['steelblue','coral','mediumseagreen'], edgecolor='white')
axes[1,2].set_title('Furnishing Status vs Avg Price')
axes[1,2].set_xlabel('Furnishing Status')
axes[1,2].set_ylabel('Average Price')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA plots saved → 'eda_visualizations.png'")


# ─────────────────────────────────────────────────────────────
# STEP 6: FEATURE SELECTION
# ─────────────────────────────────────────────────────────────
print("\n🎯 STEP 6: Feature Selection")

feature_cols = ['area', 'bedrooms', 'bathrooms', 'stories',
                'mainroad', 'guestroom', 'basement', 'hotwaterheating',
                'airconditioning', 'parking', 'prefarea', 'furnishingstatus']

X = df_encoded[feature_cols]
y = df_encoded['price']

print(f"   Features : {feature_cols}")
print(f"   Target   : price")
print(f"   X shape  : {X.shape} | y shape: {y.shape}")

# Correlation with price
print("\n--- Feature Correlation with Price (sorted) ---")
corr_target = df_encoded.corr()['price'].drop('price').sort_values(ascending=False)
print(corr_target.round(4))


# ─────────────────────────────────────────────────────────────
# STEP 7: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n✂️  STEP 7: Train / Test Split (80% / 20%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")


# ─────────────────────────────────────────────────────────────
# STEP 8: FEATURE SCALING
# ─────────────────────────────────────────────────────────────
print("\n⚖️  STEP 8: Feature Scaling (StandardScaler)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("✅ Scaling done — mean=0, std=1")


# ─────────────────────────────────────────────────────────────
# STEP 9: TRAIN THE MODEL
# ─────────────────────────────────────────────────────────────
print("\n🤖 STEP 9: Training Linear Regression Model...")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("✅ Model trained successfully!")


# ─────────────────────────────────────────────────────────────
# STEP 10: EVALUATE THE MODEL
# ─────────────────────────────────────────────────────────────
print("\n📈 STEP 10: Model Evaluation")

y_pred_train = model.predict(X_train_scaled)
y_pred_test  = model.predict(X_test_scaled)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
r2_train   = r2_score(y_train, y_pred_train)
r2_test    = r2_score(y_test,  y_pred_test)

print(f"\n   {'Metric':<28} {'Train':>12} {'Test':>12}")
print(f"   {'-'*54}")
print(f"   {'RMSE (lower = better)':<28} {rmse_train:>12.0f} {rmse_test:>12.0f}")
print(f"   {'R² Score (higher = better)':<28} {r2_train:>12.4f} {r2_test:>12.4f}")

print(f"""
📝 Interpretation:
   • RMSE of ₹{rmse_test:,.0f} → predictions off by this amount on average
   • R² of {r2_test:.4f} → model explains {r2_test*100:.1f}% of price variance
   • {'✅ Good fit!' if r2_test > 0.6 else '⚠️  Model needs improvement'}
""")


# ─────────────────────────────────────────────────────────────
# STEP 11: INTERPRET COEFFICIENTS
# ─────────────────────────────────────────────────────────────
print("\n🔎 STEP 11: Feature Coefficients")

coeff_df = pd.DataFrame({
    'Feature'    : feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\n   Intercept : ₹{model.intercept_:,.0f}")
print(f"\n   {'Feature':<20} {'Coefficient':>15}")
print(f"   {'-'*37}")
for _, row in coeff_df.iterrows():
    direction = "↑ increases price" if row['Coefficient'] > 0 else "↓ decreases price"
    print(f"   {row['Feature']:<20} {row['Coefficient']:>12,.0f}   {direction}")

# Coefficient bar chart
plt.figure(figsize=(10, 6))
colors = ['steelblue' if c > 0 else 'tomato' for c in coeff_df['Coefficient']]
plt.barh(coeff_df['Feature'], coeff_df['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Coefficients — Linear Regression', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient Value (Impact on Price)')
plt.tight_layout()
plt.savefig('feature_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved → 'feature_coefficients.png'")


# ─────────────────────────────────────────────────────────────
# STEP 12: ACTUAL vs PREDICTED PLOT
# ─────────────────────────────────────────────────────────────
print("\n📉 STEP 12: Actual vs Predicted Plot")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color='steelblue', s=30, label='Predictions')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Price (₹)')
plt.ylabel('Predicted Price (₹)')
plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved → 'actual_vs_predicted.png'")


# ─────────────────────────────────────────────────────────────
# STEP 13: SAVE THE MODEL
# ─────────────────────────────────────────────────────────────
print("\n💾 STEP 13: Saving Model & Scaler")

joblib.dump(model,  'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Saved → 'house_price_model.pkl'")
print("✅ Saved → 'scaler.pkl'")


# ─────────────────────────────────────────────────────────────
# STEP 14: EXAMPLE PREDICTIONS ON NEW HOUSES
# ─────────────────────────────────────────────────────────────
print("\n🏠 STEP 14: Example Predictions on New Houses")

loaded_model  = joblib.load('house_price_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Columns: area, bedrooms, bathrooms, stories, mainroad, guestroom,
#          basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus
new_houses = pd.DataFrame([
    [7000, 4, 2, 2,  1, 1, 1, 0, 1, 2, 1, 2],   # Big luxury house
    [4000, 3, 1, 1,  1, 0, 0, 0, 0, 1, 0, 1],   # Medium house
    [2000, 2, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0],   # Small basic house
], columns=feature_cols)

new_scaled   = loaded_scaler.transform(new_houses)
predictions  = loaded_model.predict(new_scaled)

labels = ["Luxury House (7000 sqft, AC, furnished)",
          "Medium House (4000 sqft, semi-furnished)",
          "Basic  House (2000 sqft, unfurnished)"]

print(f"\n   {'House':<42} {'Predicted Price':>18}")
print(f"   {'-'*62}")
for label, pred in zip(labels, predictions):
    print(f"   {label:<42} ₹{pred:>16,.0f}")


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ PROJECT COMPLETE — Final Summary")
print("=" * 60)
print(f"""
  Dataset      : Housing.csv (Kaggle) — 545 houses
  Features     : {len(feature_cols)} (numeric + encoded categorical)
  Algorithm    : Linear Regression
  Train/Test   : 80% ({X_train.shape[0]} samples) / 20% ({X_test.shape[0]} samples)

  📊 Results:
  ├── RMSE (Test)  : ₹{rmse_test:,.0f}
  └── R²   (Test)  : {r2_test:.4f}  ({r2_test*100:.1f}% variance explained)

  📁 Saved Files:
  ├── house_price_model.pkl     ← Trained model
  ├── scaler.pkl                ← Feature scaler
  ├── eda_visualizations.png   ← EDA charts
  ├── feature_coefficients.png ← Feature impact chart
  └── actual_vs_predicted.png  ← Prediction accuracy chart

  🚀 Next Steps:
  ├── 1. Push to GitHub → Syntecxhub_House_Price_Prediction
  └── 2. Submit via Syntecxhub Submission Form
""")
print("=" * 60)
