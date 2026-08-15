# CQROS Model Catalog

Version: 1.0.0

Status: Active

---

# 1. Purpose

This document defines every machine learning model
supported by CQROS.

It specifies

- Intended use
- Input requirements
- Training procedure
- Validation protocol
- Hyperparameters
- Deployment requirements
- Monitoring expectations

Models may only be deployed if they satisfy the
requirements defined in this document.

---

# 2. Model Philosophy

CQROS follows a research-first approach.

Models should be

- Interpretable where practical
- Robust
- Reproducible
- Versioned
- Continuously evaluated

No model is deployed solely because it performs well
on a single backtest.

---

# 3. Model Categories

Statistical Models

Tree-Based Models

Linear Models

Deep Learning Models

Time-Series Models

Probabilistic Models

Ensemble Models

Meta Models

---

# 4. Common Requirements

Every model requires

- Version
- Feature list
- Training dataset
- Validation dataset
- Random seed
- Training metadata
- Evaluation report

---

# 5. Logistic Regression

Category

Linear Model

Best For

- Binary classification
- Probability estimation
- Baseline model

Strengths

- Fast
- Interpretable
- Stable

Weaknesses

- Linear assumptions
- Limited nonlinear learning

---

# 6. Random Forest

Category

Tree Ensemble

Best For

- Baseline nonlinear modeling
- Feature importance

Strengths

- Robust
- Handles nonlinear relationships

Weaknesses

- Larger models
- Slower inference

---

# 7. XGBoost

Category

Gradient Boosting

Best For

- Structured tabular data

Strengths

- Excellent predictive accuracy
- Handles missing values
- Mature ecosystem

Weaknesses

- Hyperparameter sensitive

---

# 8. LightGBM

Category

Gradient Boosting

Recommended For

Large feature spaces

Strengths

- Extremely fast
- Memory efficient
- Excellent for tabular finance data

Deployment Status

Preferred production model.

---

# 9. CatBoost

Category

Gradient Boosting

Strengths

- Strong categorical handling
- Good default parameters

Weaknesses

- Larger training time

---

# 10. Linear Regression

Purpose

Regression baseline

Typical Use

Forecasting continuous returns.

---

# 11. Elastic Net

Combines

- L1
- L2

Useful when

Feature selection is important.

---

# 12. Support Vector Machine

Recommended only for

Small datasets.

Not recommended for

Large-scale production inference.

---

# 13. Hidden Markov Model

Purpose

Market regime detection.

Outputs

- Bull
- Bear
- Range
- High Volatility

Used by

Strategy Engine

Risk Engine

---

# 14. Gaussian Mixture Model

Purpose

Unsupervised clustering

Applications

- Market segmentation
- Regime discovery

---

# 15. K-Means

Applications

- Market clustering
- Regime exploration

Research only.

---

# 16. Isolation Forest

Applications

- Outlier detection
- Data quality
- Market anomaly detection

---

# 17. LSTM

Category

Deep Learning

Purpose

Sequential modeling

Strengths

Captures long-term dependencies.

Weaknesses

Expensive training.

---

# 18. GRU

Alternative to LSTM.

Lower computational cost.

---

# 19. Temporal CNN

Purpose

Sequential pattern recognition

Advantages

Highly parallel training.

---

# 20. Transformer

Purpose

Long-range sequence modeling

Applications

- Price forecasting
- Representation learning

Research status

Experimental.

---

# 21. Autoencoder

Purpose

Representation learning

Applications

- Compression
- Anomaly detection
- Latent feature extraction

---

# 22. Ensemble Models

Supported

Voting

Stacking

Blending

Weighted averaging

Ensembles require diversity among base models.

---

# 23. Meta Models

Purpose

Combine

- Multiple signals
- Multiple strategies
- Multiple model outputs

Example

Gradient boosting over signal probabilities.

---

# 24. Hyperparameter Optimization

Supported

- Bayesian Optimization
- Random Search
- Grid Search

Preferred

Bayesian Optimization.

---

# 25. Cross Validation

Supported

Walk-forward validation

Rolling windows

Expanding windows

Random K-Fold is prohibited for time-series.

---

# 26. Evaluation Metrics

Classification

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Log Loss

Regression

- RMSE
- MAE
- MAPE
- R²

Trading

- Sharpe
- Sortino
- Calmar
- Profit Factor
- Win Rate
- Max Drawdown
- Expectancy

---

# 27. Feature Importance

Supported

- SHAP
- Gain
- Permutation
- Split Count

Every production model must expose an explainability
report.

---

# 28. Model Registry

Each model records

- Model ID
- Version
- Training data
- Feature version
- Metrics
- Author
- Timestamp
- Git commit

---

# 29. Promotion Criteria

Models must satisfy

- Validation complete
- No leakage detected
- Performance exceeds baseline
- Stability verified
- Risk approved
- Registered in Model Registry

---

# 30. Monitoring

Production monitoring includes

- Prediction latency
- Drift detection
- Feature drift
- Data quality
- Performance degradation

---

# 31. Retraining

Retraining may occur

- Scheduled
- Drift-triggered
- Research-triggered

Every retrained model receives a new version.

---

# 32. Reproducibility

Store

- Random seed
- Dependency versions
- Configuration
- Training dataset
- Feature version

Training must be reproducible.

---

# 33. Security

Models are immutable after registration.

Artifacts include

- Checksums
- Signatures
- Metadata

---

# 34. Future Models

Potential additions

- TabNet
- FT-Transformer
- N-BEATS
- Temporal Fusion Transformer
- Graph Neural Networks
- Diffusion-based forecasting

---

# 35. Summary

CQROS supports a diverse collection of statistical,
tree-based, probabilistic, and deep learning models.

Every model follows standardized training, validation,
registration, monitoring, and deployment procedures,
ensuring reproducible research and safe production use.