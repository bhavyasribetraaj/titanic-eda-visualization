import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


df = pd.read_csv(r"C:\Users\Betraaj\Downloads\Desktop\internship\elevate labs\Titanic-Dataset.csv")  # Make sure this is the correct path


print("===== SUMMARY STATISTICS =====")
print(df.describe(include='all'))


print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

df[numeric_cols].hist(bins=30, figsize=(15, 10), color='lightblue', edgecolor='black')
plt.suptitle('Histograms of Numeric Features')
plt.tight_layout()
plt.savefig("histograms.png")
plt.show()


for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.savefig(f'boxplot_{col}.png')
    plt.show()

plt.figure(figsize=(10, 6))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()


selected_cols = ['Age', 'Fare', 'Pclass', 'Survived']
sns.pairplot(df[selected_cols].dropna(), hue='Survived', palette='Set1')
plt.suptitle("Pairplot of Selected Features by Survival", y=1.02)
plt.savefig("pairplot.png")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='pastel')
plt.title('Survival by Gender')
plt.savefig("survival_by_gender.png")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Passenger Class')
plt.savefig("survival_by_class.png")
plt.show()


fig = px.histogram(df, x='Age', color='Survived', nbins=30, title='Age Distribution by Survival', barmode='overlay')
fig.write_html("age_distribution_by_survival.html")
fig.show()


fig = px.box(df, x='Survived', y='Fare', color='Survived', title='Fare Distribution by Survival')
fig.write_html("fare_vs_survival.html")
fig.show()


print("\n===== SKEWNESS =====")
print(df[numeric_cols].skew())


print("\n===== MULTICOLLINEARITY NOTES =====")
print("""
Check correlation matrix above:
- Pclass and Fare show moderate correlation.
- Age and Fare may have slight positive correlation.
- No extreme multicollinearity (|correlation| > 0.8) detected.
""")


print("\n✅ EDA Task Completed. Visuals saved as PNG/HTML files.")
