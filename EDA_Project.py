import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("D:/SEM 4/INT375/ProjectData.csv")
print(df.info())
print(df.describe())

# Check for missing values
print("\nMissing values in each column:\n", df.isna().sum())

# Check column names
print("Columns:", df.columns.tolist())

# Chart 1: Histogram - Net Area Sown
plt.figure(figsize=(8, 5))
sns.histplot(df['Net area sown'], bins=40, kde=True, color='skyblue')
plt.title('Distribution of Net Area Sown')
plt.xlabel('Net Area Sown')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Chart 2: Bar Plot - Mean Net Area Cultivated by Category of Holdings
plt.figure(figsize=(10, 5))
mean_values = df.groupby('Category of holdings')['Net area cultivated'].mean().sort_values(ascending=False)
sns.barplot(x=mean_values.index, y=mean_values.values, palette='viridis')
plt.title('Average Net Area Cultivated by Category of Holdings')
plt.ylabel('Mean Net Area Cultivated')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 3: Barplot - Mean Net Area Sown by State
state_avg = df.groupby('srcStateName')['Net area sown'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=state_avg.values, y=state_avg.index, palette='viridis')
plt.title('Top 10 States by Mean Net Area Sown')
plt.xlabel('Mean Net Area Sown')
plt.ylabel('State')
plt.tight_layout()
plt.show()

# Chart 4: Pie Chart - Distribution of Category of Holdings
cat_counts = df['Category of holdings'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Land Holdings Categories')
plt.tight_layout()
plt.show()

# Chart 5: Scatterplot - Net Area Sown vs Net Area Cultivated
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Net area sown', y='Net area cultivated')
plt.title('Net Area Sown vs Net Area Cultivated')
plt.tight_layout()
plt.show()

# Chart 6: Correlation Heatmap
num_cols = ['Net area sown', 'Area under current fallows', 'Net area cultivated', 'Uncultivated area ']
corr = df[num_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
            linecolor='gray', square=True, cbar_kws={'shrink': 0.7})
plt.title('Correlation Heatmap of Area Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Chart 7: Horizontal bar plots by state
columns = ['Net area sown', 'Area under current fallows', 'Net area cultivated', 'Uncultivated area ']
category = 'srcStateName'
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(columns):
    sns.barplot(
        data=df.sort_values(by=col, ascending=False).head(25),
        x=col,
        y=category,
        ax=axes[i],
        hue=category,
        palette='rocket'
    )
    axes[i].set_title(f'Top 10 by {col}')
    axes[i].set_xlabel(f'{col} Value')
    axes[i].set_ylabel('')
plt.tight_layout()
plt.show()

# Regression: Predict Net Area Cultivated
x = df[['Net area sown']]
y = df[['Net area cultivated']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

# Prediction example
check = pd.DataFrame({'Net area sown': [5000]})
result = model.predict(check)
print("Predicted Net Area Cultivated for 5000 Net Area Sown:", result[0][0])

# Plot regression line
plt.scatter(x, y, color="green")
plt.plot(x, model.predict(x), color='red', linewidth=3)
plt.xlabel("Net area sown")
plt.ylabel("Net area cultivated")
plt.title("Linear Regression Fit")
plt.tight_layout()
plt.show()

# Mean Squared Error
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
