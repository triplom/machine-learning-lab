# Resources

## Source Courses

| Course | Description |
|--------|-------------|
| Machine Learning A-Z | Comprehensive ML course covering all classic algorithms in Python and R |
| TensorFlow + Deep Learning with Python | ANN, CNN, RNN, transfer learning, model deployment with Keras/TensorFlow 2 |

## Cross-references (python_learning)

| Module | Topic | Relevance |
|--------|-------|-----------|
| [11-machine-learning](https://github.com/triplom/python_learning/tree/main/11-machine-learning) | Scikit-learn pipelines, supervised + unsupervised models | Overlaps modules 01–04, 09–10 |
| [12-deep-learning](https://github.com/triplom/python_learning/tree/main/12-deep-learning) | Keras/TensorFlow, MNIST, CNNs | Overlaps module 08 |
| [13-nlp](https://github.com/triplom/python_learning/tree/main/13-nlp) | NLP with spaCy, TF-IDF search | Overlaps module 07 |

## Official Documentation

| Library | Docs |
|---------|------|
| scikit-learn | https://scikit-learn.org/stable/user_guide.html |
| TensorFlow / Keras | https://www.tensorflow.org/guide |
| XGBoost | https://xgboost.readthedocs.io |
| mlxtend | http://rasbt.github.io/mlxtend/ |
| NLTK | https://www.nltk.org/ |
| SciPy | https://docs.scipy.org/doc/scipy/ |

## Recommended Reading

| Topic | Resource |
|-------|---------|
| Scikit-learn preprocessing | https://scikit-learn.org/stable/modules/preprocessing.html |
| Bias-Variance tradeoff | https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html |
| Cross Validation guide | https://scikit-learn.org/stable/modules/cross_validation.html |
| Keras Sequential API | https://www.tensorflow.org/guide/keras/sequential_model |
| Transfer Learning | https://www.tensorflow.org/guide/keras/transfer_learning |
| XGBoost parameters | https://xgboost.readthedocs.io/en/stable/parameter.html |

## Datasets Used in This Repo

| Dataset | Module | Source |
|---------|--------|--------|
| Data.csv | 01 | Machine Learning A-Z (synthetic) |
| Salary_Data.csv | 02 | Machine Learning A-Z |
| 50_Startups.csv | 02 | Machine Learning A-Z |
| Position_Salaries.csv | 02 | Machine Learning A-Z |
| Social_Network_Ads.csv | 03 | Machine Learning A-Z |
| Mall_Customers.csv | 04 | Machine Learning A-Z |
| Wine dataset | 09 | sklearn built-in |
| Breast Cancer dataset | 10 | sklearn built-in |
| CIFAR-10 | 08 | tf.keras.datasets (auto-download) |

## Python Environment

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow xgboost mlxtend nltk scipy
python -c "import nltk; nltk.download('stopwords')"
```
