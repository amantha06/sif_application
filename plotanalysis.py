import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

features_df = pd.read_csv('features.csv')
test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')

print("Features Data:")
print(features_df.head())

print("\nTest Data:")
print(test_df.head())

print("\nTrain Data:")
print(train_df.head())

features_df['Date'] = pd.to_datetime(features_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])
train_df['Date'] = pd.to_datetime(train_df['Date'])

print("\nSummary Statistics for Features Data:")
print(features_df.describe())

print("\nSummary Statistics for Test Data:")
print(test_df.describe())

print("\nSummary Statistics for Train Data:")
print(train_df.describe())

merged_df = pd.merge(train_df, features_df, on=['Store', 'Date'], how='left')

plt.figure(figsize=(14, 6))
sns.lineplot(data=train_df, x='Date', y='Weekly_Sales', hue='IsHoliday', palette='coolwarm')
plt.title('Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))
sns.lineplot(data=features_df, x='Date', y='Fuel_Price', color='green')
plt.title('Fuel Price Over Time')
plt.xlabel('Date')
plt.ylabel('Fuel Price')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))
sns.lineplot(data=features_df, x='Date', y='Unemployment', color='red')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.grid(True)
plt.show()

holiday_sales = train_df[train_df['IsHoliday'] == True]['Weekly_Sales']
non_holiday_sales = train_df[train_df['IsHoliday'] == False]['Weekly_Sales']

plt.figure(figsize=(10, 6))
sns.boxplot(data=[holiday_sales, non_holiday_sales], notch=True)
plt.xticks([0, 1], ['Holiday Sales', 'Non-Holiday Sales'])
plt.title('Impact of Holidays on Weekly Sales')
plt.ylabel('Weekly Sales')
plt.show()
