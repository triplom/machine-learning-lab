# Machine Learning Lab — Study Plan

An 8-week hands-on curriculum covering the full ML pipeline from data preprocessing to model deployment.

---

## Phase 1 — Foundations (Week 1)

**Goal:** Clean and prepare real-world data for ML pipelines.

| Task | Module | File |
|------|--------|------|
| Handle missing values | 01-data-preprocessing | data_preprocessing.py |
| Encode categorical data | 01-data-preprocessing | data_preprocessing.py |
| Feature scaling | 01-data-preprocessing | data_preprocessing.py |
| Train/test split | 01-data-preprocessing | data_preprocessing.py |
| Exercises | 01-data-preprocessing | exercises/ |

**Cross-reference:** [python_learning/11-machine-learning](https://github.com/triplom/python_learning/tree/main/11-machine-learning) — sklearn overview

---

## Phase 2 — Supervised Learning: Regression (Week 2)

**Goal:** Predict continuous values using 6 regression algorithms.

| Algorithm | File |
|-----------|------|
| Simple Linear Regression | 02-regression/simple_linear_regression.py |
| Multiple Linear Regression | 02-regression/multiple_linear_regression.py |
| Polynomial Regression | 02-regression/polynomial_regression.py |
| Support Vector Regression | 02-regression/svr.py |
| Decision Tree Regression | 02-regression/decision_tree_regression.py |
| Random Forest Regression | 02-regression/random_forest_regression.py |

**Key Datasets:** Salary_Data.csv, 50_Startups.csv, Position_Salaries.csv

---

## Phase 3 — Supervised Learning: Classification (Week 3)

**Goal:** Predict categories using 7 classification algorithms.

| Algorithm | File |
|-----------|------|
| Logistic Regression | 03-classification/logistic_regression.py |
| K-Nearest Neighbors | 03-classification/knn.py |
| Support Vector Machine | 03-classification/svm.py |
| Kernel SVM | 03-classification/kernel_svm.py |
| Naive Bayes | 03-classification/naive_bayes.py |
| Decision Tree | 03-classification/decision_tree_classification.py |
| Random Forest | 03-classification/random_forest_classification.py |

**Key Dataset:** Social_Network_Ads.csv

---

## Phase 4 — Unsupervised Learning + Reinforcement Learning (Week 4)

**Goal:** Find hidden structure in data; solve bandit problems.

| Topic | Module | File |
|-------|--------|------|
| K-Means Clustering | 04-clustering | kmeans.py |
| Hierarchical Clustering | 04-clustering | hierarchical.py |
| Apriori Association Rules | 05-association-rules | apriori.py |
| Upper Confidence Bound (UCB) | 06-reinforcement-learning | ucb.py |
| Thompson Sampling | 06-reinforcement-learning | thompson.py |

**Key Dataset:** Mall_Customers.csv

---

## Phase 5 — NLP (Week 5)

**Goal:** Process text, build bag-of-words model, classify sentiment.

| Topic | File |
|-------|------|
| Text cleaning & tokenization | 07-nlp/text_classification.py |
| Bag of words (CountVectorizer) | 07-nlp/text_classification.py |
| Naive Bayes text classifier | 07-nlp/text_classification.py |

**Cross-reference:** [python_learning/13-nlp](https://github.com/triplom/python_learning/tree/main/13-nlp) — advanced NLP with spaCy

---

## Phase 6 — Deep Learning (Weeks 6–7)

**Goal:** Build and train neural networks with TensorFlow/Keras.

| Topic | File |
|-------|------|
| Artificial Neural Network (ANN) | 08-deep-learning/ann.py |
| Convolutional Neural Network (CNN) | 08-deep-learning/cnn.py |
| Model saving and loading | 08-deep-learning/ann.py |
| Transfer learning | 08-deep-learning/cnn.py |

**Cross-reference:** [python_learning/12-deep-learning](https://github.com/triplom/python_learning/tree/main/12-deep-learning) — Keras MNIST basics

---

## Phase 7 — Dimensionality Reduction (Week 8, Part 1)

**Goal:** Reduce feature space for better performance and visualization.

| Algorithm | File |
|-----------|------|
| Principal Component Analysis (PCA) | 09-dimensionality-reduction/pca.py |
| Linear Discriminant Analysis (LDA) | 09-dimensionality-reduction/lda.py |
| Kernel PCA | 09-dimensionality-reduction/kernel_pca.py |

---

## Phase 8 — Model Selection & Boosting (Week 8, Part 2)

**Goal:** Tune and compare models systematically.

| Topic | File |
|-------|------|
| k-Fold Cross Validation | 10-model-selection/cross_validation.py |
| Grid Search (hyperparameter tuning) | 10-model-selection/grid_search.py |
| XGBoost | 10-model-selection/xgboost_lab.py |

---

## Weekly Schedule

| Week | Focus | Estimated Hours |
|------|-------|----------------|
| 1 | Data Preprocessing | 4–6 h |
| 2 | Regression | 6–8 h |
| 3 | Classification | 6–8 h |
| 4 | Clustering + RL | 5–7 h |
| 5 | NLP | 4–6 h |
| 6 | Deep Learning — ANN | 6–8 h |
| 7 | Deep Learning — CNN | 6–8 h |
| 8 | Dim. Reduction + Model Selection | 5–7 h |

**Total:** ~45–60 hours

---

## Cheatsheets

| Topic | File |
|-------|------|
| Preprocessing | [cheatsheets/sklearn-preprocessing.md](../cheatsheets/sklearn-preprocessing.md) |
| Regression | [cheatsheets/regression.md](../cheatsheets/regression.md) |
| Classification | [cheatsheets/classification.md](../cheatsheets/classification.md) |
| Clustering | [cheatsheets/clustering.md](../cheatsheets/clustering.md) |
| Deep Learning / Keras | [cheatsheets/deep-learning-keras.md](../cheatsheets/deep-learning-keras.md) |
| Model Selection | [cheatsheets/model-selection.md](../cheatsheets/model-selection.md) |
