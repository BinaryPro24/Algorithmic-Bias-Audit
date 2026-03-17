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
