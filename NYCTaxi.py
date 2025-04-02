import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandasql import sqldf


df = pd.read_parquet('yellow_tripdata_2025-01.parquet', engine='pyarrow')
df = df.sample(n=100000, random_state=42)  # Random sample of 100k rows
print(df.head())  # Check the first few rows
print(df.columns)  # See column names

# Data Cleaning
#Inspect the Data
print(df.info())  # Data types and non-null counts
print(df.isnull().sum())  # Missing values per column

#Handle Missing Values
df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'fare_amount', 'trip_distance'])
print(f"Rows after dropping missing values: {len(df)}")

#Convert Timestamps
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

#Calculate Trip Duration
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60


#Filter Outliers
df = df[(df['fare_amount'] > 0) & (df['trip_duration'] > 0) & (df['trip_duration'] < 1440)]  # 1440 min = 24 hr
print(f"Rows after filtering outliers: {len(df)}")


#Data Analysis
#Extract Hour of Day
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

#SQL Query for Trends

query = """
SELECT pickup_hour, 
       AVG(fare_amount) as avg_fare, 
       AVG(trip_duration) as avg_duration, 
       AVG(passenger_count) as avg_passengers
FROM df 
GROUP BY pickup_hour
"""
hourly_stats = sqldf(query)
print(hourly_stats.head())


print(df[['fare_amount', 'trip_duration', 'passenger_count']].describe())

#Visualization
#Bar Chart: Average Fare by Hour

plt.figure(figsize=(10, 6))
sns.barplot(x='pickup_hour', y='avg_fare', data=hourly_stats, color='skyblue')
plt.title('Average Fare by Hour of Day')
plt.xlabel('Hour (0-23)')
plt.ylabel('Average Fare ($)')
plt.savefig('fare_by_hour.png')  # Save the plot
plt.show()


#Histogram: Trip Duration Distribution

plt.figure(figsize=(10, 6))
plt.hist(df['trip_duration'], bins=50, range=(0, 60), color='lightgreen', edgecolor='black')
plt.title('Distribution of Trip Durations (0-60 Minutes)')
plt.xlabel('Trip Duration (Minutes)')
plt.ylabel('Frequency')
plt.savefig('trip_duration_dist.png')
plt.show()


#Optional Predictive Modeling
#Prepare Data

model_df = df[['trip_distance', 'passenger_count', 'fare_amount']].dropna()
X = model_df[['trip_distance', 'passenger_count']]  # Features
y = model_df['fare_amount']  # Target

#Train a Linear Regression Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")























































