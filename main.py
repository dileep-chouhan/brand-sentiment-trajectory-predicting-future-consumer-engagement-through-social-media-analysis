import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=365))
sentiment = 0.5 + 0.2 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 0.1, 365) # Seasonal trend + noise
engagement = 100 + 50 * sentiment + 10 * np.random.normal(0, 1, 365) # Engagement depends on sentiment
marketing_spend = 50 + 20 * np.random.normal(0,1,365) #random marketing spend
df = pd.DataFrame({'Date': dates, 'Sentiment': sentiment, 'Engagement': engagement, 'MarketingSpend':marketing_spend})
# --- 2. Data Cleaning and Feature Engineering ---
#No cleaning needed for synthetic data.  Adding a week of the year feature.
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
# --- 3. Analysis ---
#Correlation Analysis
correlation_matrix = df[['Sentiment', 'Engagement', 'MarketingSpend']].corr()
print("Correlation Matrix:\n", correlation_matrix)
#Regression Analysis (Engagement vs Sentiment)
slope, intercept, r_value, p_value, std_err = linregress(df['Sentiment'], df['Engagement'])
print(f"\nLinear Regression (Engagement vs Sentiment):")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")
print(f"P-value: {p_value:.3f}")
# --- 4. Visualization ---
#Sentiment Trajectory
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Sentiment', data=df)
plt.title('Brand Sentiment Trajectory')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('sentiment_trajectory.png')
print("Plot saved to sentiment_trajectory.png")
#Engagement vs Sentiment Scatter Plot with Regression Line
plt.figure(figsize=(12,6))
sns.regplot(x='Sentiment', y='Engagement', data=df)
plt.title('Engagement vs Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Engagement')
plt.grid(True)
plt.tight_layout()
plt.savefig('engagement_vs_sentiment.png')
print("Plot saved to engagement_vs_sentiment.png")
#Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Plot saved to correlation_heatmap.png")