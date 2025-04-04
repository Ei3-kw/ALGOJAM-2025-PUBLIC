import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Load the datasets
fried_chicken = pd.read_csv('data/Fried Chicken_price_history.csv')
raw_chicken = pd.read_csv('data/Raw Chicken_price_history.csv')
secret_spices = pd.read_csv('data/Secret Spices_price_history.csv')

# Merge the datasets on 'Day'
df = pd.merge(fried_chicken, raw_chicken, on='Day', suffixes=('_fried', '_raw'))
df = pd.merge(df, secret_spices, on='Day', suffixes=('', '_spices'))
df.columns = ['Day', 'Price_fried', 'Price_raw', 'Price_spices']

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# Prepare data for regression
X = df[['Price_raw', 'Price_spices']]
y = df['Price_fried']

# Add constant for the intercept
X_with_const = sm.add_constant(X)

# Fit the linear model
model = sm.OLS(y, X_with_const).fit()
print("Linear Model Summary:")
print(model.summary().tables[1])  # Just show coefficients table for brevity

# Calculate predictions and residuals
y_pred_linear = model.predict(X_with_const)
residuals = y - y_pred_linear

# Add residuals to dataframe
df['Residuals'] = residuals

# Calculate RMSE for the linear model
rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
print(f"RMSE (Linear Model): {rmse_linear:.6f}")

# Try different EMA spans on the residuals
spans = range(3,69)
results = []

plt.figure(figsize=(14, 8))
plt.subplot(211)
plt.plot(df['Day'], residuals, label='Original Residuals', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Original Residuals from Linear Model')
plt.grid(True, alpha=0.3)

plt.subplot(212)
for span in spans:
    df[f'EMA_{span}'] = calculate_ema(df['Residuals'], span)
    
    # combined model: Linear prediction + EMA of residuals
    df[f'Prediction_EMA_{span}'] = y_pred_linear + df[f'EMA_{span}'].shift(1)
    
    # Calculate RMSE (excluding first row because of the shift)
    rmse_combined = np.sqrt(mean_squared_error(
        y[1:], df[f'Prediction_EMA_{span}'][1:]))
    
    results.append({
        'EMA_Span': span,
        'RMSE': rmse_combined
    })
    
    # Plot the EMA
    plt.plot(df['Day'], df[f'EMA_{span}'], label=f'EMA (span={span})', alpha=0.69)

plt.legend()
plt.title('EMA of Residuals with Different Spans')
plt.grid(True, alpha=0.420)
plt.tight_layout()

# Find the best EMA span
results_df = pd.DataFrame(results)
best_span = results_df.loc[results_df['RMSE'].idxmin(), 'EMA_Span']
best_rmse = results_df['RMSE'].min()

print(f"\nBest EMA Span: {best_span}")
print(f"RMSE with EMA correction: {best_rmse:.6f}")
print(f"Improvement over linear model: {(rmse_linear - best_rmse) / rmse_linear * 100:.2f}%")

# Visualize the best combined model
best_column = f'Prediction_EMA_{int(best_span)}'

plt.figure(figsize=(14, 10))

# Plot actual vs linear model
plt.subplot(311)
plt.plot(df['Day'], y, 'r-', alpha=0.69, label='Actual Fried Chicken Price')
plt.plot(df['Day'], y_pred_linear, 'c-', alpha=0.69, label='Linear Model Prediction')
plt.title('Actual vs Linear Model Prediction')
plt.legend()
plt.grid(True, alpha=0.420)

# Plot actual vs combined model
plt.subplot(312)
plt.plot(df['Day'], y, 'r-', alpha=0.69, label='Actual Fried Chicken Price')
plt.plot(df['Day'][1:], df[best_column][1:], 'c-', alpha=0.69, label='Combined Model Prediction')
plt.title(f'Actual vs Combined Model (Linear + EMA{int(best_span)})')
plt.legend()
plt.grid(True, alpha=0.420)

# Plot residuals from both models
plt.subplot(313)
plt.plot(df['Day'], residuals, 'r-', alpha=0.69, label='Linear Model Residuals')
plt.plot(df['Day'][1:], y[1:] - df[best_column][1:], 'c-', alpha=0.69, label='Combined Model Residuals')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.420)
plt.title('Comparison of Residuals')
plt.legend()
plt.grid(True, alpha=0.420)

plt.tight_layout()

# Create the final prediction model
print("\nFinal Combined Model:")
print(f"1. Base prediction = {model.params[0]:.4f} + {model.params[1]:.4f} × Raw Chicken Price + {model.params[2]:.4f} × Secret Spices Price")
print(f"2. Apply EMA(span={int(best_span)}) to the residuals")
print(f"3. Final prediction = Base prediction + EMA of residuals")

plt.show()

# Create a function that can be used for future predictions
def predict_fried_chicken_price(raw_price, spice_price, previous_residuals):
    # Calculate base prediction from linear model
    base_pred = model.params[0] + model.params[1] * raw_price + model.params[2] * spice_price
    
    # Calculate EMA of previous residuals
    residuals_series = pd.Series(previous_residuals)
    ema_residual = calculate_ema(residuals_series, int(best_span)).iloc[-1]
    
    # Combined prediction
    return base_pred + ema_residual

