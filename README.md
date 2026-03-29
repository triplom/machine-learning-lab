# Machine Learning Lab

A hands-on, lab-driven machine learning curriculum covering classical ML, deep learning, NLP, and model deployment. Built from two source courses and cross-referenced with [python_learning](https://github.com/triplom/python_learning).

## Source Courses

| # | Course | Content |
|---|--------|---------|
| 1 | Machine Learning A-Z | Data preprocessing, regression, classification, clustering, association rules, RL, NLP, deep learning, dimensionality reduction, model selection |
| 2 | TensorFlow + Deep Learning with Python | ANNs, CNNs, RNNs, transfer learning, model saving/deployment with Keras/TensorFlow |

## Cross-References (python_learning)

| Module | Topic | Link |
|--------|-------|------|
| 11 | Scikit-learn models (supervised + unsupervised) | [11-machine-learning](https://github.com/triplom/python_learning/tree/main/11-machine-learning) |
| 12 | Keras / TensorFlow deep learning | [12-deep-learning](https://github.com/triplom/python_learning/tree/main/12-deep-learning) |
| 13 | NLP — text search & recommenders | [13-nlp](https://github.com/triplom/python_learning/tree/main/13-nlp) |

## Repo Structure

```
machine-learning-lab/
├── study-plan/                    # Phase-by-phase ML curriculum
├── 01-data-preprocessing/         # Missing data, encoding, scaling, train/test split
├── 02-regression/                 # Simple, Multiple, Polynomial, SVR, DT, RF
├── 03-classification/             # Logistic, KNN, SVM, Kernel SVM, Naive Bayes, DT, RF
├── 04-clustering/                 # K-Means, Hierarchical
├── 05-association-rules/          # Apriori
├── 06-reinforcement-learning/     # UCB, Thompson Sampling
├── 07-nlp/                        # Bag of words, text classification
├── 08-deep-learning/              # ANNs, CNNs with TensorFlow/Keras
├── 09-dimensionality-reduction/   # PCA, LDA, Kernel PCA
├── 10-model-selection/            # k-Fold CV, Grid Search, XGBoost
├── cheatsheets/                   # Quick-reference markdown sheets
├── resources/                     # Course links and learning resources
└── legacy/                        # Original forked files (Python, R, templates)
```

## Quick Start

```bash
git clone https://github.com/triplom/machine-learning-lab.git
cd machine-learning-lab
pip install numpy pandas matplotlib scikit-learn tensorflow xgboost mlxtend
```

## Prerequisites

- Python 3.9+
- Familiarity with Python basics (see [python_learning modules 01–05](https://github.com/triplom/python_learning))
- NumPy and Pandas basics (see [python_learning modules 08–09](https://github.com/triplom/python_learning))

## Related Projects

| Project | Relationship |
|---------|-------------|
| [python_learning](https://github.com/triplom/python_learning) | **Prerequisite** — Python foundations (Modules 01–05), NumPy/Pandas (10), Scikit-learn (11), Keras/TF (12), NLP (13) must be covered before or alongside this lab |
| [kali-hacking-lab](https://github.com/triplom/kali-hacking-lab) | **Shared skill set** — ML anomaly detection and NLP techniques apply to threat intelligence and log analysis (Month 3 Vulnerability Analysis, Month 9 Sniffing) |
| [LearningGamification](https://github.com/triplom/LearningGamification) | **Applied ML opportunity** — the gamification engine's point scoring and course recommendations can be extended with regression/classification/NLP models from this lab |
| [lpic3-study-lab](https://github.com/triplom/lpic3-study-lab) | **Infrastructure for ML** — LPIC-3 305 Virtualization covers containerising Python/ML workloads with Docker; 701 DevOps covers CI/CD pipelines for model deployment |

## Roadmap

See [study-plan/README.md](study-plan/README.md) for the full phase-by-phase curriculum.

| Phase | Modules | Weeks |
|-------|---------|-------|
| 1 — Foundations | 01 (preprocessing) | 1 |
| 2 — Supervised Learning | 02 (regression), 03 (classification) | 2–3 |
| 3 — Unsupervised + RL | 04 (clustering), 05 (assoc. rules), 06 (RL) | 4 |
| 4 — NLP | 07 | 5 |
| 5 — Deep Learning | 08 (ANN + CNN) | 6–7 |
| 6 — Advanced | 09 (dim. reduction), 10 (model selection) | 8 |
