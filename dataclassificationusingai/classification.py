# ============================================
# Project 2: Data Classification Using AI
# Dataset: Iris (built-in from sklearn)
# Algorithm: Decision Tree Classifier
# ============================================

# --- 1. Import Libraries ---
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import os

# --- 2. Load and Understand Dataset ---
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df['target'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

print("=" * 45)
print("        IRIS DATASET - OVERVIEW")
print("=" * 45)
print(df.head(10))
print(f"\nTotal Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
print("\nClass Distribution:")
print(df['flower_name'].value_counts())
print("\nBasic Statistics:")
print(df.describe())

# --- 3. Split Data ---
X = iris.data   # Features (input)
y = iris.target # Labels (output)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 80% train, 20% test
    random_state=42
)

print("\n" + "=" * 45)
print("           DATA SPLIT SUMMARY")
print("=" * 45)
print(f"Total Samples     : {len(X)}")
print(f"Training Samples  : {len(X_train)} (80%)")
print(f"Testing Samples   : {len(X_test)} (20%)")

# --- 4. Train the Model ---
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

# --- 5. Make Predictions & Evaluate ---
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

print("\n" + "=" * 45)
print("           MODEL EVALUATION")
print("=" * 45)
print(f"Accuracy: {accuracy:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# --- 6. Visualization 1: Decision Tree ---
os.makedirs("output", exist_ok=True)

plt.figure(figsize=(14, 7))
tree.plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    fontsize=10
)
plt.title("Decision Tree - Iris Flower Classification", fontsize=14)
plt.savefig("output/decision_tree.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Decision tree saved → output/decision_tree.png")

# --- 7. Visualization 2: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("output/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved → output/confusion_matrix.png")

# --- 8. Visualization 3: Feature Importance ---
importances = model.feature_importances_
feat_names = iris.feature_names

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feat_names, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.savefig("output/feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance saved → output/feature_importance.png")

print("\n" + "=" * 45)
print("       PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 45)