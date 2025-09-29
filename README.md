# Decision Trees & Random Forests Project

## Objective

Learn tree-based models for classification & regression using the Heart Disease dataset.

## Tools Used

* Python 3.x
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Graphviz

## Dataset

Heart Disease Dataset (CSV format)

* Features include patient data such as age, sex, blood pressure, cholesterol, etc.
* Target: `0` (No Disease), `1` (Disease)

## Project Structure

```
Decision-Trees-Random-Forests/
│
├─ data/
│   └─ heart.csv          # Dataset
│
├─ outputs/
│   ├─ decision_tree_alt.png   # Alternative decision tree plot
│   └─ feature_importances.png # Random forest feature importance plot
│
├─ src/
│   └─ decision_tree_rf.py # Main script for training and evaluation
│
└─ README.md
```

## Steps Implemented

1. **Load Dataset**: Read CSV using Pandas.
2. **Train-Test Split**: Split data into 80% train and 20% test.
3. **Decision Tree Classifier**: Train a DT, evaluate accuracy, precision, recall, f1-score.
4. **Visualize the Tree**: Render with Graphviz and save as PNG. (Alternative plot available if Graphviz fails)
5. **Pruned Decision Tree**: Limit max_depth to control overfitting.
6. **Random Forest Classifier**: Train RF, evaluate, and compare accuracy with DT.
7. **Feature Importance**: Plot feature importances using Seaborn barplot.
8. **Cross-validation**: Evaluate Random Forest with 5-fold CV.

## Key Results

* Decision Tree Accuracy: ~0.985
* Random Forest Accuracy: 1.0
* Cross-validation Mean Accuracy: ~0.997
* Feature importance plot saved to `outputs/feature_importances.png`

## Notes

* Graphviz must be installed and accessible via PATH for tree rendering.
* If Graphviz fails, an alternative Matplotlib plot is saved as `outputs/decision_tree_alt.png`.
