import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import graphviz

# ------------------------------
# Ensure outputs folder exists
# ------------------------------
os.makedirs("outputs", exist_ok=True)

# ------------------------------
# 1. Load Dataset
# ------------------------------
data = pd.read_csv(r"C:\Users\harsh\python-projects\Decision-Trees-Random-Forests\data\heart.csv")
print("Dataset shape:", data.shape)
print("Columns:", data.columns)


X = data.drop("target", axis=1)
y = data["target"]

# ------------------------------
# 2. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 3. Decision Tree Classifier
# ------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n📌 Decision Tree Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# ------------------------------
# 4a. Visualize the Tree (Graphviz)
# ------------------------------
try:
    dot_data = export_graphviz(
        dt,
        out_file=None,
        feature_names=X.columns,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("outputs/decision_tree_graphviz", format="png", cleanup=True)
    print("✅ Graphviz tree saved at outputs/decision_tree_graphviz.png")
except Exception as e:
    print("⚠️ Graphviz visualization failed:", e)
    print("You can still see alternative plot in outputs/decision_tree_alt.png")

# ------------------------------
# 4b. Alternative Tree Plot (Matplotlib)
# ------------------------------
plt.figure(figsize=(20,10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree - Full (Matplotlib)")
plt.savefig("outputs/decision_tree_alt.png")
plt.close()

# ------------------------------
# 5. Overfitting Control (max_depth)
# ------------------------------
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_pruned = dt_pruned.predict(X_test)

print("\n📌 Pruned Decision Tree (max_depth=4) Accuracy:", accuracy_score(y_test, y_pred_pruned))

# ------------------------------
# 6. Random Forest Classifier
# ------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n📌 Random Forest Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ------------------------------
# 7. Feature Importances
# ------------------------------
importances = rf.feature_importances_
feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

# ------------------------------
# 8. Cross Validation
# ------------------------------
cv_scores = cross_val_score(rf, X, y, cv=5)
print("\n📌 Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
