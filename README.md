# Algorithmic Bias Audit 🔍

## Overview
An audit of the UCI Adult Income dataset identifying gender and race-based 
income disparities that would be replicated by any machine learning model 
trained on this data without bias mitigation.

Built in the context of **EU AI Act (2024)** compliance, which mandates 
bias auditing for high-risk AI systems used in financial services.

## Key Findings

![Bias Audit Chart](bias_audit_chart.png)

- **Gender gap**: Men earn >$50k at 30.6% vs women at 10.9%, a **3x disparity**
- **Race gap**: White and Asian-Pacific Islander earners significantly outperform 
  Black, Native American and Other groups — up to a **2.8x disparity**
- A loan approval model trained on this data would systematically 
  disadvantage women and minority ethnic applicants; not by design
  but by inheritance from historical discrimination

## Why This Matters
Under the EU AI Act, any high-risk AI system used in banking, credit scoring 
or employment must be audited for bias before deployment. Companies that fail 
to comply face fines of up to €30 million.

This project demonstrates the first step of that audit process; identifying 
disparities in training data before a model is even built.

## Methodology
- Loaded UCI Adult Income dataset (32,561 records, 15 features)
- Analysed income distribution across protected characteristics
- Calculated high income rates by gender and race
- Visualised disparities against dataset average baseline

## Tech Stack
- Python 3.9
- Pandas
- Matplotlib
- NumPy
