# Predicting AI Model Adoption on Hugging Face
This project investigates whether AI model adoption on open platforms can be predicted using only information visible at browse time, and which signals matter most when developers choose between competing models.

Using a large sample of Hugging Face models, I apply supervised learning and model interpretability techniques to examine how creator reputation, model metadata, and temporal factors relate to adoption outcomes.

[Read the full paper (PDF)](research-paper.pdf)

## Research Question

> To what extent can AI model adoption be predicted from browse-time data, and which observable features most strongly influence adoption on open-source platforms?

This question is motivated by discovery costs and information asymmetry on large platforms, where users face thousands of alternatives but limited ability to evaluate true quality ex ante.

# Data
Source: Hugging Face models api

Final sample: 45,707 models

**Outcome variable:**

- is_high_adoption — binary indicator for models in the top 25% of downloads

Features engineered from browse-time information only:

- Creator productivity (n_models_by_author)
- Author type (individual, organization, AI company, big tech)
- Model family, size, fine-tuning status
- Model age (days since last update)
- Domain and language 

# Methods
- Naïve Bayes:
Used as a baseline classifier to test whether adoption signals operate additively under independence assumptions.

- XGBoost (Gradient Boosting):
Used to capture non-linearities and feature interactions, with class imbalance handled via weighting.

- Model Evaluation:
AUC, Precision, Recall, F1

- SHAP values: used to explain global and local feature contributions

# Key Findings
- Adoption is predictable from browse-time signals alone (XGBoost AUC ≈ 0.83).
- Performance gains over Naïve Bayes indicate that feature interactions matter, not just additive effects.
- Creator reputation and cumulative exposure dominate prediction, outweighing most technical descriptors.
- Author productivity and model age are the strongest predictors.
- Institutional backing (e.g. big tech) improves adoption, but much of its effect operates through correlated signals like contributor track record.
- Results suggest a self-reinforcing diffusion process, where visibility reduces uncertainty and attracts further adoption.

## Why This Project Matters
This project demonstrates how platform-level signals shape technology diffusion, even in technically sophisticated environments. It combines real-world, messy platform data with interpretations aligned with business and platform insights. The approach is applicable to marketplaces, creator platforms, open-source ecosystems, and product discovery problems.
