# Algorithmic Bias Audit 🔍

## Overview
A full bias audit pipeline applied to the UCI Adult Income dataset. It's identifying gender and race-based disparities in raw data and then proving those disparities 
by amplifying by a machine learning model.

Built in the context of **EU AI Act (2024)** compliance,  which mandates bias auditing for high-risk AI systems used in financial services.

## Key Findings

![Bias Audit Chart](bias_audit_chart.png)

### Raw Data Disparities
- **Gender gap**: Men earn >$50k at 30.6% vs women at 10.9% — a **3x disparity.**

- **Race gap**: White and Asian-Pacific Islander earners significantly outperform.

  Black, Native American and Other groups, up to a **2.8x disparity.**

### Model Bias
After training a Logistic Regression classifier on this data:

- **Male False Negative Rate: 56.4%**
- **Female False Negative Rate: 92.3%**

When a woman actually earns over $50k, the model gets it wrong **92.3% of the time**, predicting she earns less than she does.

If a bank used this model for loan approvals, a high-earning woman would be systematically denied or underserved; not because of her finances but 
because of her demographics.

## Race False Negative Rates
- **White: 60.7%**
- **Asian-Pacific Islander: 61.3%**
- **Other: 66.7%**
- **Black: 78.5%**
- **Amer-Indian-Eskimo: 90.0%**

## The Compounding Effect
A high-earning Black or Native American woman faces bias from two directions 
simultaneously; gender and race false negative rates stacking on top of each 
other. In a loan approval context, that's not just unfair but it's a legal 
liability under the EU AI Act.

This compounding effect is the most dangerous form of algorithmic bias because it's invisible without a full intersectional audit.

## Why This Matters
Under the EU AI Act, any high-risk AI system used in banking, credit scoring or employment must be audited for bias before deployment. Companies that fail to comply face fines of up to €30 million.

This project demonstrates a complete audit pipeline; from identifying disparities in raw training data to proving a trained model replicates and amplifies those disparities.

### Fairlearn Audit - Disparity Scores
Using the Fairlearn library to quantify bias formally:

**Gender Disparity Score: 0.359**
**Race Disparity Score: 0.293**

A disparity score above 0.1 is considered significant in fairness research.
The gender disparity of 0.359 is more than 3x the acceptable threshold, meaning this model would formally **fail** a fairness audit in a regulated financial services context.

This is the exact metric compliance teams use to determine whether an AI system is fit for deployment under the EU AI Act.

## Tech Stack
- Python 3.9
- Pandas
- Matplotlib
- Scikit-learn
- Fairlearn