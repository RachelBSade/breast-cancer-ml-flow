"""
Part 1(c): - Learning Problem and dataset explaination.
This assignment focuses on a binary classification problem where the goal is to predict
if a breast tumor is Benign (0) or Malignant (1).
The dataset contains 30 numerical features that describe different properties of the tumor,
such as radius, texture, concavity, symmetry, and fractal dimension.
These features are provided in three versions: mean, standard error, and worst values.
The dataset is already split into training and testing sets, so no additional split is needed. For evaluation,
we use the F1 score, which balances precision and recall, making it suitable for this type of medical data
where both false positives and false negatives are important to minimize.
"""

## Part 2 - Initial Preparations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer

train_path = "C:/Users/rache/Desktop/machine learning/cancer_train.csv"
test_path  = "C:/Users/rache/Desktop/machine learning/cancer_test.csv"
y_col = "target"
drop_cols = []
feature_cols = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", "radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"]

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

for c in drop_cols:
    if c in df_train.columns: df_train = df_train.drop(columns=[c])
    if c in df_test.columns:  df_test  = df_test.drop(columns=[c])

X_train = df_train[feature_cols].copy()
y_train = df_train[y_col].astype(str).copy()

X_test = df_test[feature_cols].copy()
y_test = df_test[y_col].astype(str).copy() if y_col in df_test.columns else None

print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
print("\nTrain label distribution:")
print(y_train.value_counts())

display(X_train.head())
display(X_test.head())

# EDA charts (matplotlib only)
plt.figure()
y_train.value_counts().plot(kind='bar', rot=0, title='Class Balance (Train)')
plt.xlabel('Class'); plt.ylabel('Count'); plt.tight_layout(); plt.show()

# 2) Distributions of a few features
some_feats = feature_cols[:3] if len(feature_cols) >= 3 else feature_cols
for feat in some_feats:
    plt.figure()
    X_train[feat].plot(kind='hist', bins=30, title=f'Distribution of {feat}')
    plt.xlabel(feat); plt.ylabel('Frequency'); plt.tight_layout(); plt.show()

# 3) Correlation heatmap (top 12 variance features)
variances = X_train.var().sort_values(ascending=False)
top_feats = variances.index[:12].tolist()
corr = X_train[top_feats].corr()

plt.figure(figsize=(6,5))
im = plt.imshow(corr, interpolation='nearest')
plt.title('Correlation Heatmap (Top 12 Var Features)')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(top_feats)), top_feats, rotation=90)
plt.yticks(range(len(top_feats)), top_feats)
plt.tight_layout(); plt.show()

## Part 3 - Experiments
def make_preprocess(scale: bool):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps)

def make_selector(k):
    return 'passthrough' if k is None else SelectKBest(score_func=f_classif, k=k)

pipe = Pipeline(steps=[
    ('pre', make_preprocess(False)),
    ('sel', 'passthrough'),
    ('clf', LogisticRegression(max_iter=200, random_state=42))
])

param_grid = [
    {
        'pre': [make_preprocess(False), make_preprocess(True)],
        'sel': [make_selector(None), make_selector(10)],
        'clf': [LogisticRegression(max_iter=200, solver='liblinear', random_state=42)],
        'clf__C': [0.1, 1.0, 10.0],
    },
    {
        'pre': [make_preprocess(False), make_preprocess(True)],
        'sel': [make_selector(None), make_selector(10)],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [None, 10],
    },
]

majority_label = y_train.value_counts().idxmax()
scorer = make_scorer(f1_score, pos_label=majority_label)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=False,
    verbose=0
)

grid.fit(X_train, y_train)

print("Best CV F1 (majority class):", grid.best_score_)
print("Best params:")
print(grid.best_params_)

res = pd.DataFrame(grid.cv_results_)
res_simple = pd.DataFrame({
    'rank_test_score': res['rank_test_score'],
    'mean_test_f1_majority': res['mean_test_score'],
    'param_pre': res['param_pre'].astype(str),
    'param_sel': res['param_sel'].astype(str),
    'param_clf': res['param_clf'].astype(str),
    'param_clf_C': res.get('param_clf_C', pd.Series([np.nan]*len(res))),
    'param_clf_n_estimators': res.get('param_clf_n_estimators', pd.Series([np.nan]*len(res))),
    'param_clf_max_depth': res.get('param_clf_max_depth', pd.Series([np.nan]*len(res))),
}).sort_values('rank_test_score')

display(res_simple.head(10))
res_simple.to_csv('cv_results_final.csv', index=False)
print("Saved grid results cv_results_final.csv")

## Part 4 - Training
best_model = grid.best_estimator_
print(best_model)

## Part 5 - Apply on test and show model performance estimation
if y_test is not None:
    y_pred = best_model.predict(X_test)
    majority_label = y_train.value_counts().idxmax()
    f1_majority = f1_score(y_test, y_pred, pos_label=majority_label)
    print(f"Test F1 (majority class='{majority_label}'): {f1_majority:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    labels_order = [majority_label] + [l for l in sorted(y_train.unique()) if l != majority_label]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    import matplotlib.pyplot as plt
    plt.figure()
    disp.plot(values_format='d')
    plt.title('Confusion Matrix (Test)')
    plt.tight_layout(); plt.show()

    out_df = pd.DataFrame({'y_true': y_test.reset_index(drop=True),
                           'y_pred': pd.Series(y_pred).reset_index(drop=True)})
    display(out_df.head(5))
    out_df.to_csv(r'C:\Users\rache\Desktop\machine learning\test_predictions_final_sample.csv', index=False)
    print("Saved sample predictions to test_predictions_final_sample.csv")
else:
    print("No labels found in test set; skipping evaluation.")
    ##