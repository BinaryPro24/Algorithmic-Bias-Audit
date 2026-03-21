import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fairlearn.metrics import MetricFrame, false_negative_rate
from sklearn.metrics import accuracy_score

# Load the UCI Adult Income dataset directly from the web
# This dataset contains census data used to predict income levels
# It's widely used in algorithmic fairness research
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Define column names; the dataset has no header row
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'income'
]

# Load data and strip whitespace from string columns
df = pd.read_csv(url, names=columns, skipinitialspace=True)

print(df.shape)
print(df.head())
print("\n--- INCOME DISTRIBUTION ---")
print(df['income'].value_counts())
print("\n--- GENDER DISTRIBUTION ---")
print(df['gender'].value_counts())
print("\n--- RACE DISTRIBUTION ---")
print(df['race'].value_counts())

# Analyse income distribution by gender
print("\n--- INCOME BY GENDER ---")
gender_income = df.groupby(['gender', 'income']).size().unstack()
gender_income['high_income_rate'] = (
    gender_income['>50K'] / 
    (gender_income['<=50K'] + gender_income['>50K']) * 100
).round(1)
print(gender_income)

# Analyse income distribution by race
print("\n--- INCOME BY RACE ---")
race_income = df.groupby(['race', 'income']).size().unstack()
race_income['high_income_rate'] = (
    race_income['>50K'] / 
    (race_income['<=50K'] + race_income['>50K']) * 100
).round(1)
print(race_income)

# VISUALISATION: Income Disparity by Gender and Race in the UCI Adult Income Dataset
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Algorithmic Bias Audit — UCI Adult Income Dataset', 
             fontsize=14, fontweight='bold')

# Gender chart
genders = ['Male', 'Female']
rates = [30.6, 10.9]
colors = ['royalblue', 'coral']
bars1 = ax1.bar(genders, rates, color=colors, width=0.5)
ax1.set_title('High Income Rate by Gender')
ax1.set_ylabel('% Earning >$50k')
ax1.set_ylim(0, 40)
ax1.axhline(y=24.1, color='red', linestyle='--', label='Dataset average (24.1%)')
ax1.legend()
for bar, rate in zip(bars1, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{rate}%', ha='center', fontweight='bold')

# Race chart
races = ['White', 'Asian-Pac\nIslander', 'Black', 'Amer-Indian\nEskimo', 'Other']
race_rates = [25.6, 26.6, 12.4, 11.6, 9.2]
colors2 = ['royalblue', 'royalblue', 'coral', 'coral', 'coral']
bars2 = ax2.bar(races, race_rates, color=colors2, width=0.5)
ax2.set_title('High Income Rate by Race')
ax2.set_ylabel('% Earning >$50k')
ax2.set_ylim(0, 35)
ax2.axhline(y=24.1, color='red', linestyle='--', label='Dataset average (24.1%)')
ax2.legend()
for bar, rate in zip(bars2, race_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{rate}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('bias_audit_chart.png')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Preparing data for modelling 
# Convert income to binary: 1 = earns >50k, 0 = earns <=50k
df['income_binary'] = (df['income'] == '>50K').astype(int)

# Select features for the model
features = ['age', 'education_num', 'hours_per_week', 'gender', 'race']
df_model = df[features + ['income_binary']].copy()

# Encode categorical variables (gender, race) into numbers
le_gender = LabelEncoder()
le_race = LabelEncoder()
df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'])
df_model['race_encoded'] = le_race.fit_transform(df_model['race'])

# Final feature set
X = df_model[['age', 'education_num', 'hours_per_week', 
              'gender_encoded', 'race_encoded']]
y = df_model['income_binary']

# Split into training and test sets
# 80% of data used to train the model, 20% reserved for testing
# random_state = 42 ensures reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# StandardScaler normalises features to the same scale
# Prevents age (0-90) from dominating hours_per_week (0-99)
# Essential for Logistic Regression to work correctly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression predicts binary outcomes (high income vs low income)
# max_iter = 1000 gives the model enough steps to converge on a solution
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("\n--- MODEL PERFORMANCE ---")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# --- BIAS MEASUREMENT BY GENDER ---
print("\n--- FALSE NEGATIVE RATE BY GENDER ---")
X_test_full = X_test.copy()
X_test_full['actual'] = y_test.values
X_test_full['predicted'] = y_pred
X_test_full['gender'] = df_model.loc[X_test.index, 'gender'].values

#  False Negative = model predicted <=50k but person actually earns >50k
# A high FNR for women means the model systematically underestimates 
# female earning potential, critical failure in a loan approval context
for gender in ['Male', 'Female']:
    group = X_test_full[X_test_full['gender'] == gender]
    actual_positive = group[group['actual'] == 1]
    false_negatives = actual_positive[actual_positive['predicted'] == 0]
    fnr = len(false_negatives) / len(actual_positive) * 100
    print(f"{gender}: False Negative Rate = {fnr:.1f}%")

# Bias measurement by race
print("\n--- FALSE NEGATIVE RATE BY RACE ---")
X_test_full['race'] = df_model.loc[X_test.index, 'race'].values

for race in df['race'].unique():
    group = X_test_full[X_test_full['race'] == race]
    actual_positive = group[group['actual'] == 1]
    if len(actual_positive) > 0:
        false_negatives = actual_positive[actual_positive['predicted'] == 0]
        fnr = len(false_negatives) / len(actual_positive) * 100
        print(f"{race}: False Negative Rate = {fnr:.1f}%")

# Fairlearn Bias Audit
print("\n--- FAIRLEARN AUDIT BY GENDER ---")

# Create a MetricFrame to measure false negative rate by gender
gender_metric = MetricFrame(
    metrics=false_negative_rate,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test_full['gender']
)

print("False Negative Rate by Gender: ")
print(gender_metric.by_group)
print(f"\nOverall False Negative Rate: {gender_metric.overall:.3f}")
print(f"Disparity (max - min): {gender_metric.difference():.3f}")


print("\n--- FAIRLEARN AUDIT BY RACE ---")

# Create a MetricFrame to measure false negative rate by race
race_metric = MetricFrame(
    metrics=false_negative_rate,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test_full['race']
)

print("False Negative Rate by Race:")
print(race_metric.by_group)
print(f"\nOverall False Negative Rate: {race_metric.overall:.3f}")
print(f"Disparity (max - min): {race_metric.difference():.3f}")