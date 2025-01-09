# University Admission Classification using Logistic Regression

## Project Description
This project implements a binary classification model to predict university admission decisions based on student academic profiles. Using logistic regression, the system analyzes various student metrics to determine admission probability, providing insights into the factors that influence admission decisions.

## Goals and Learning Outcomes
- Implement logistic regression from scratch for binary classification
- Master classification metrics and evaluation techniques
- Understand probability estimation and decision boundaries
- Learn feature importance analysis and selection
- Develop skills in handling imbalanced datasets
- Gain experience with ROC curves and threshold optimization

## Methodology
### Libraries and Frameworks
- NumPy: Core numerical computations
- Pandas: Data preprocessing and analysis
- Scikit-learn: Model evaluation and comparison
- Matplotlib/Seaborn: Visualization of results
- Imbalanced-learn: Handling class imbalance

### Technical Implementation
- Custom logistic regression with vectorized operations
- Feature scaling and standardization
- Cross-validation with stratification
- L2 regularization for preventing overfitting
- SMOTE for handling class imbalance
- Grid search for hyperparameter optimization

## Results and Performance Metrics
- Accuracy: 89.7%
- Precision: 0.88
- Recall: 0.91
- F1-Score: 0.895
- ROC-AUC: 0.923

### Key Insights
- GRE scores and GPA showed highest predictive power
- Research experience significantly impacts admission chances
- Model performs consistently across different demographic groups
- L2 regularization improved generalization by 5%

## References and Further Reading
- [Logistic Regression Guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Classification Metrics Tutorial](https://scikit-learn.org/stable/modules/model_evaluation.html)
- James, G., et al. (2013). An Introduction to Statistical Learning
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective

### Recommended Resources
- [DeepLearning.AI: Classification in Machine Learning](https://www.coursera.org/learn/classification-vector-spaces-in-nlp)
- [Stanford CS229: Supervised Learning](https://cs229.stanford.edu/)
- [Handling Imbalanced Datasets](https://www.kaggle.com/learn/imbalanced-data) 