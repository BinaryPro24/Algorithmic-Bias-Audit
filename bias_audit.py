import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model on scaled data
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

for gender in ['Male', 'Female']:
    group = X_test_full[X_test_full['gender'] == gender]
    actual_positive = group[group['actual'] == 1]
    false_negatives = actual_positive[actual_positive['predicted'] == 0]
    fnr = len(false_negatives) / len(actual_positive) * 100
    print(f"{gender}: False Negative Rate = {fnr:.1f}%")