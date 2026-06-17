# Project 2: Data Classification Using AI

## Overview
This project builds a basic classification model using the Iris dataset
and a Decision Tree algorithm to classify flowers into 3 categories.

## Dataset
- **Name:** Iris Dataset (built-in from scikit-learn)
- **Total Samples:** 150
- **Features:** Sepal length, Sepal width, Petal length, Petal width
- **Classes:** Setosa, Versicolor, Virginica (50 samples each)

## Algorithm Used
- **Decision Tree Classifier**

## Steps Followed
1. Loaded and explored the Iris dataset
2. Split data into 80% training and 20% testing
3. Trained a Decision Tree Classifier
4. Evaluated model performance
5. Visualized results using graphs

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-Score | 1.00 |

## Output Files
- `output/decision_tree.png` - Decision Tree visualization
- `output/confusion_matrix.png` - Confusion Matrix
- `output/feature_importance.png` - Feature Importance chart

## Libraries Used
- pandas
- scikit-learn
- matplotlib
- seaborn

## How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
python classification.py
```

## Key Learnings
- How to load and explore a dataset
- How to split data into training and testing sets
- How to train a classification model
- How to evaluate model accuracy
- How to visualize results