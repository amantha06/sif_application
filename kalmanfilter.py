import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import cProfile

train_df = pd.read_csv('train.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])

sales_series = train_df.set_index('Date')['Weekly_Sales']
print(f"Number of data points: {len(sales_series)}")

missing_values = sales_series.isnull().sum()
print(f"Number of missing values: {missing_values}")

if missing_values > 0:
    sales_series = sales_series.fillna(method='ffill')

kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
try:
    cProfile.run('state_means, _ = kf.em(sales_series.values, n_iter=5).smooth(sales_series.values)')
    plt.figure(figsize=(14, 6))
    plt.plot(sales_series.index, sales_series, label='Observed Sales', color='blue', alpha=0.6)
    plt.plot(sales_series.index, state_means, label='Smoothed Sales (Kalman Filter)', color='red', linestyle='--')
    plt.title('Kalman Filter Applied to Weekly Sales Data')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"An error occurred while running the Kalman filter: {e}")

sales_subset = sales_series.head(1000)  
try:
    state_means_subset, _ = kf.em(sales_subset.values, n_iter=5).smooth(sales_subset.values)
    plt.figure(figsize=(14, 6))
    plt.plot(sales_subset.index, sales_subset, label='Observed Sales (Subset)', color='blue', alpha=0.6)
    plt.plot(sales_subset.index, state_means_subset, label='Smoothed Sales (Subset)', color='green', linestyle='--')
    plt.title('Kalman Filter Applied to Subset of Weekly Sales Data')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"An error occurred while running the Kalman filter on the subset: {e}")
