# additional_sales_analysis.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Increase plot size for better readability
plt.rcParams['figure.figsize'] = [12, 6]

# 1. Data Loading and Preprocessing
# ---------------------------------
# Load the data
train_df = pd.read_csv('train.csv')
features_df = pd.read_csv('features.csv')

# Merge the datasets on 'Store', 'Date', and 'IsHoliday'
train_df['Date'] = pd.to_datetime(train_df['Date'])
features_df['Date'] = pd.to_datetime(features_df['Date'])
data = pd.merge(train_df, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')

# Handle missing values
data.fillna(0, inplace=True)

# Convert 'IsHoliday' to integer for numerical analysis
data['IsHoliday'] = data['IsHoliday'].astype(int)

# Extract additional time-related features
data['DayOfWeek'] = data['Date'].dt.day_name()
data['DayOfWeekNum'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month_name()
data['MonthNum'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# 2. Correlation Analysis
# -----------------------
# Select numerical columns for correlation
corr_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
             'IsHoliday', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

# Compute correlation matrix
corr_matrix = data[corr_cols].corr()

# Plot heatmap
plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 3. Seasonal Decomposition
# -------------------------
# Aggregate sales data by date
sales_series = data.groupby('Date')['Weekly_Sales'].sum()

# Perform seasonal decomposition
decomposition = seasonal_decompose(sales_series, model='additive', period=52)  # Weekly data with annual seasonality

# Plot the decomposition
decomposition.plot()
plt.show()

# 4. Autocorrelation Analysis
# ---------------------------
# Plot ACF and PACF
plt.figure()
plt.subplot(121)
plot_acf(sales_series, lags=50, ax=plt.gca())
plt.subplot(122)
plot_pacf(sales_series, lags=50, ax=plt.gca())
plt.tight_layout()
plt.show()

# 5. Stationarity Testing
# -----------------------
# Perform ADF test
adf_result = adfuller(sales_series)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'   {key}: {value}')

# 6. Lagged Variable Analysis
# ---------------------------
# Aggregate sales and CPI data by date
data_agg = data.groupby('Date').agg({'Weekly_Sales':'sum', 'CPI':'mean'}).reset_index()
data_agg.set_index('Date', inplace=True)

# Cross-correlation function
def cross_corr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag))

# Calculate cross-correlations for lags -10 to 10
lags = range(-10, 11)
ccor = [cross_corr(data_agg['Weekly_Sales'], data_agg['CPI'], lag) for lag in lags]

# Plot the cross-correlation
plt.figure()
plt.bar(lags, ccor)
plt.title('Cross-Correlation between Weekly Sales and CPI')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')
plt.show()

# 7. Sales Distribution Analysis
# ------------------------------
# Plot histogram and KDE
plt.figure()
sns.histplot(data['Weekly_Sales'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of Weekly Sales')
plt.xlabel('Weekly Sales')
plt.ylabel('Frequency')
plt.show()

# 8. Department-Level Analysis
# ----------------------------
# Calculate average sales per department
dept_sales = data.groupby('Dept')['Weekly_Sales'].mean().reset_index()
dept_sales = dept_sales.sort_values('Weekly_Sales', ascending=False)

# Plot top 10 departments
plt.figure()
sns.barplot(x='Dept', y='Weekly_Sales', data=dept_sales.head(10), palette='viridis')
plt.title('Top 10 Departments by Average Weekly Sales')
plt.xlabel('Department')
plt.ylabel('Average Weekly Sales')
plt.show()

# 9. Customer Behavior Analysis
# -----------------------------
# Average sales by day of the week
sales_by_day = data.groupby('DayOfWeek')['Weekly_Sales'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

plt.figure()
sales_by_day.plot(kind='bar', color='coral')
plt.title('Average Weekly Sales by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Weekly Sales')
plt.show()

# Average sales by month
months_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
sales_by_month = data.groupby('Month')['Weekly_Sales'].mean().reindex(months_order)

plt.figure()
sales_by_month.plot(kind='bar', color='lightgreen')
plt.title('Average Weekly Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Weekly Sales')
plt.show()

# 10. Outlier Detection
# ---------------------
plt.figure()
sns.boxplot(x=data['Weekly_Sales'], color='lightblue')
plt.title('Boxplot of Weekly Sales')
plt.xlabel('Weekly Sales')
plt.show()

# 11. Enhanced Predictive Modeling
# --------------------------------
# Prepare data for modeling
features = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI',
            'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
            'DayOfWeekNum', 'MonthNum', 'Year']
X = data[features]
y = data['Weekly_Sales']

# Handle categorical variables if necessary (e.g., one-hot encoding)
# For simplicity, we'll proceed without encoding, as all features are numerical or ordinal

# Scale numerical features
scaler = StandardScaler()
X[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
   'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = scaler.fit_transform(
       X[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
          'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']]
   )

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost Model
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2 = r2_score(y_test, y_pred_xgb)
print("XGBoost Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.4f}\n")

# Feature Importance from XGBoost
importances = xgb.feature_importances_
feature_names = features
xgb_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure()
xgb_importances.plot(kind='bar')
plt.title('Feature Importance in XGBoost Model')
plt.ylabel('Importance Score')
plt.show()

# 12. Clustering Analysis
# -----------------------
# Cluster stores based on sales patterns
store_sales = data.groupby('Store')['Weekly_Sales'].mean().reset_index()
X_cluster = store_sales[['Weekly_Sales']]

# Scale the data
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
store_sales['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Plot clusters
plt.figure()
sns.scatterplot(x='Store', y='Weekly_Sales', hue='Cluster', data=store_sales, palette='Set1')
plt.title('Store Clusters Based on Average Weekly Sales')
plt.xlabel('Store')
plt.ylabel('Average Weekly Sales')
plt.legend(title='Cluster')
plt.show()

# 13. Promotional Impact Analysis
# -------------------------------
# Analyze the impact of MarkDowns on sales
promotion_data = data[['Weekly_Sales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']]

# Correlation between MarkDowns and Sales
promo_corr = promotion_data.corr()['Weekly_Sales'].drop('Weekly_Sales')
print("Correlation between MarkDowns and Weekly Sales:")
print(promo_corr)

# Plotting the correlations
plt.figure()
promo_corr.plot(kind='bar', color='teal')
plt.title('Correlation between MarkDowns and Weekly Sales')
plt.ylabel('Correlation Coefficient')
plt.show()

# 14. Data Quality Assessment
# ---------------------------
# Check for missing values after merging datasets
print("Missing values per column after merging:")
print(data.isnull().sum())

# Since we filled missing values earlier, there should be none
# However, we can check for duplicates
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# 15. Conclusion
# --------------
# This script provides additional analyses not covered in your Overleaf code, offering deeper insights
# into the sales data, customer behavior, and factors influencing sales performance.

# Save the figures (optional)
# Uncomment the following lines if you wish to save the plots as image files
# for i, fig in enumerate(plt.get_fignums()):
#     plt.figure(fig)
#     plt.savefig(f'figure_{i}.png')
