import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the datasets
fried_chicken = pd.read_csv('data/Fried Chicken_price_history.csv')
raw_chicken = pd.read_csv('data/Raw Chicken_price_history.csv')
secret_spices = pd.read_csv('data/Secret Spices_price_history.csv')

# Merge on 'Day'
df = pd.merge(fried_chicken, raw_chicken, on='Day', suffixes=('_fried', '_raw'))
df = pd.merge(df, secret_spices, on='Day', suffixes=('', '_spices'))
df.columns = ['Day', 'Price_fried', 'Price_raw', 'Price_spices']

scatter_matrix = sns.pairplot(df[['Price_fried', 'Price_raw', 'Price_spices']], 
                             height=2.5, aspect=1.5, 
                             plot_kws={'alpha': 0.69, 's': 15})
plt.suptitle('Scatter Plot Matrix of Price Relationships', y=1.02, fontsize=16)
plt.tight_layout()

# Calculate correlation coefficients
correlation_matrix = df[['Price_fried', 'Price_raw', 'Price_spices']].corr()
print(f"Correlation Matrix:\n{correlation_matrix}")

# Calculate Spearman rank correlation for non-linear relationships
spearman_corr = df[['Price_fried', 'Price_raw', 'Price_spices']].corr(method='spearman')
print(f"\nSpearman Rank Correlation Matrix:\n{spearman_corr}")

# Multiple regression analysis
# Prepare data for regression
X = df[['Price_raw', 'Price_spices']]
y = df['Price_fried']

# Add constant for the intercept
X_with_const = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X_with_const).fit()

print(f"\nRegression Results:\n{model.summary()}")

# multicollinearity?
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factors (VIF):")
print(vif_data)


# Model predictions
y_pred = model.predict(X_with_const)
model_rmse = np.sqrt(((y - y_pred) ** 2).mean())
print(f"\nRoot Mean Squared Error (RMSE): {model_rmse:.6f}")

# predictions vs actual
plt.figure(figsize=(9, 6))
plt.scatter(df['Day'], y, color='orange', label='Actual', alpha=0.69)
plt.plot(df['Day'], y_pred, color='cyan', label='Predicted')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Actual vs Predicted Fried Chicken Prices')
plt.legend()
plt.grid(True, alpha=0.420)

# Plot residuals to check for patterns
plt.figure(figsize=(9, 6))
residuals = y - y_pred
plt.scatter(df['Day'], residuals, color='orange', alpha=0.69)
plt.axhline(y=0, color='cyan', linestyle='-')
plt.xlabel('Day')
plt.ylabel('Residuals')
plt.title('Residuals Over Time')
plt.grid(True, alpha=0.420)

# Calculate contribution of each variable
# Standardised coefficients
X_std = (X - X.mean()) / X.std()
X_std = sm.add_constant(X_std)
model_std = sm.OLS(y, X_std).fit()
print(f"\nStandardised Coefficients:\n{model_std.summary().tables[1]}")

# Percentage contribution to model
importance = abs(model_std.params[1:]) / sum(abs(model_std.params[1:]))
print("\nRelative Importance of Predictors:")
for i, var in enumerate(X.columns):
    print(f"{var}: {importance[i]*100:.2f}%")

# Alternative regression using sklearn
model_sk = LinearRegression()
model_sk.fit(X, y)
r_squared = model_sk.score(X, y)
print(f"\nR-squared (sklearn): {r_squared:.4f}")
print(f"Raw Chicken coefficient: {model_sk.coef_[0]:.4f}")
print(f"Secret Spices coefficient: {model_sk.coef_[1]:.4f}")
print(f"Intercept: {model_sk.intercept_:.4f}")

print(f"\nConclusion:\nFried Chicken Price ≈ {model_sk.intercept_:.4f} + {model_sk.coef_[0]:.4f} × Raw Chicken Price + {model_sk.coef_[1]:.4f} × Secret Spices Price")

plt.show()
