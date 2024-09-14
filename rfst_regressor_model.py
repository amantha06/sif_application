import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

features_df = pd.read_csv('features.csv')
train_df = pd.read_csv('train.csv')

features_df['Date'] = pd.to_datetime(features_df['Date'])
train_df['Date'] = pd.to_datetime(train_df['Date'])
data = pd.merge(train_df, features_df, on=['Store', 'Date'], how='left')

print("Columns in merged data:", data.columns)

if 'IsHoliday' not in data.columns:
    print("Warning: 'IsHoliday' column not found in the merged DataFrame. Adding default values.")
    data['IsHoliday'] = 0
else:
    data['IsHoliday'] = data['IsHoliday'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)

data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['DayOfWeek'] = data['Date'].dt.dayofweek

features = ['Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
            'IsHoliday', 'Day', 'Month', 'Year', 'DayOfWeek']
X = data[features]
y = data['Weekly_Sales']

X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importance in Sales Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(True)
plt.show()
