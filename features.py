# import libraries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

# import data
df = pd.read_csv("/data.csv")

# split data into outcome and features
# y = df.iloc[:, 1]

y = df.iloc[:, 2]

X = df.iloc[:, 3:]

# shuffle rows
X_shuffled = X.sample(frac=1, random_state=13).reset_index(drop=True)
y_shuffled = y.sample(frac=1, random_state=13).reset_index(drop=True)

# split data into training+validation and testing sets in an 85:15 ratio
X_train_val, X_test, y_train_val, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.15, stratify=y_shuffled, random_state=13)

# split data into training and validation sets in a 75:25 ratio
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=13)

# standardize data
# initialize the scaler
scaler = RobustScaler()

# save column names and row indices
train_columns = X_train.columns
train_index = X_train.index
val_columns = X_val.columns
val_index = X_val.index
test_columns = X_test.columns
test_index = X_test.index

# fit the scaler to the training data
scaler.fit(X_train)

# transform the data using the RobustScaler
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# convert the data back to pandas dfs
X_train = pd.DataFrame(data=X_train, columns=train_columns, index=train_index)
X_val = pd.DataFrame(data=X_val, columns=val_columns, index=val_index)
X_test = pd.DataFrame(data=X_test, columns=test_columns, index=test_index)

# remove redundant features with low variance
# create a variance threshold object
# selector = VarianceThreshold(threshold=(0.05))

# fit the variance threshold selector to the features df
# selector.fit(X_train)

# create a boolean mask to keep features with variance above the threshold
# mask = selector.get_support(indices=False)

# apply the mask to create a new pd df
# X_train_var = X_train.loc[:, mask]

# print the original and reduced shapes
# print(f"Original shape: {X_train.shape}")
# print(f"Reduced shape: {X_train_var.shape}")

# skip variance thresholding
# X_train_var = X_train

# remove redundant features with high correlation
# compute the Spearman correlation matrix and keep only absolute values
# correlation_matrix = X_train_var.corr(method="spearman").abs()

# create a new df, upper, containing the upper triangular portion of the correlation matrix excluding the diagonal
# upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# create a list which contains names of columns that >=1 correlation value greater than 0.8
# to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# create a new df which drops the columns listed in the to_drop list, removing highly correlated columns
# X_train_cor = X_train_var.drop(X_train_var[to_drop], axis=1)

# print the original and reduced shapes
# print(f"Original shape: {X_train_var.shape}")
# print(f"Reduced shape: {X_train_cor.shape}")

# use recursive feature elimination with 5-fold cross validation to select and keep only optimal features
# create a Random Forest Classifier model
# model = RandomForestClassifier(n_estimators=100, random_state=13)

# create a StratifiedKFold (5-fold) object for the cross-validation within the RFECV
# cv = StratifiedKFold(n_splits=5)

# create an RFECV object using the Random Forest model as the estimator
# rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy')

# fit the RFECV model on the features and the outcome to select the best features
# rfecv = rfecv.fit(X_train_cor, y_train)

# retrieve the column names of the selected features based on the RFECV support mask
# selected_features = X_train_cor.columns[rfecv.support_]

# create a new DataFrame dfftred2 containing only the selected features from dfftred2
# X_train_rfe = X_train_cor[selected_features]

# print the optimal number of features
# print(f"Optimal number of features: {rfecv.n_features_}")

# print the original and reduced shapes
# print(f"Original shape: {X_train_cor.shape}")
# print(f"Reduced shape: {X_train_rfe.shape}")

# save selected features
# X_train_rfe.to_csv("X_train_rfe.csv")

# directly import selected features (saves time if you want to fine tune hyperparams later on)
X_train_rfe_with_index = pd.read_csv("X_train_rfe.csv")

# remove the first index column
X_train_rfe = X_train_rfe_with_index.iloc[:, 1:]

# define metrics to quantify model performance
metrics = {
    "Accuracy": accuracy_score,
    "Precision/PPV": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=1),
    "NPV": lambda y_true, y_pred: npv(confusion_matrix(y_true, y_pred)),
    "Recall/Sensitivity": recall_score,
    "Specificity": lambda y_true, y_pred: specificity(confusion_matrix(y_true, y_pred)),
    "F1 Score": f1_score,
    "AUC-ROC": lambda y_true, y_prob: roc_auc_score(y_true, y_prob)
}

def specificity(conf_matrix):
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    return TN / (TN + FP)

def npv(conf_matrix):
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]
    return TN / (TN + FN)

# define a bootstrap function to compute 95% confidence intervals for the metrics
def bootstrap_ci(y_true, y_pred, y_pred_proba=None, metric_func=None, n_bootstrap=2000, alpha=0.05):
    bootstrap_estimates = []

    y_true = y_true.values if isinstance(y_true, (pd.DataFrame, pd.Series)) else y_true
    y_pred = y_pred.values if isinstance(y_pred, (pd.DataFrame, pd.Series)) else y_pred
    if y_pred_proba is not None:
        y_pred_proba = y_pred_proba.values if isinstance(y_pred_proba, (pd.DataFrame, pd.Series)) else y_pred_proba

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        if y_pred_proba is not None:
            y_pred_proba_sample = y_pred_proba[indices]
            estimate = metric_func(y_true_sample, y_pred_proba_sample)
        else:
            estimate = metric_func(y_true_sample, y_pred_sample)
        bootstrap_estimates.append(estimate)

    sorted_estimates = np.sort(bootstrap_estimates)
    lower = np.percentile(sorted_estimates, 100 * alpha / 2.)
    upper = np.percentile(sorted_estimates, 100 * (1 - alpha / 2.))
    return lower, upper

# create a random forest classifier 
clfrfc = RandomForestClassifier(random_state=13)

# train the classifier
clfrfc.fit(X_train_rfe, y_train)

# prediction using the random forest model on test data
y_val_score = clfrfc.predict_proba(X_val_rfe)
y_val_pred = clfrfc.predict(X_val_rfe)

# compute and print the confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
print(f"Confusion Matrix: \n{conf_matrix}")

# compute other metrics with 95% CIs
for name, func in metrics.items():
    if name == "AUC-ROC":
        point_estimate = func(y_val, y_val_score[:, 1]) # only pass probabilities for the positive class
        lower, upper = bootstrap_ci(y_val, y_val_pred, y_val_score[:, 1], func) # only pass probabilities for the positive class
    else:
        point_estimate = func(y_val, y_val_pred)
        lower, upper = bootstrap_ci(y_val, y_val_pred, None, func)
    print(f"{name}: {point_estimate:.4f} [95% CI: {lower:.4f}, {upper:.4f}]")

    # Create an XGBoost classifier
clfxgb = XGBClassifier(random_state=13)

# Train the classifier
clfxgb.fit(X_train_rfe, y_train)

# Make predictions using the XGBoost model on validation data
y_val_score = clfxgb.predict_proba(X_val_rfe)
y_val_pred = clfxgb.predict(X_val_rfe)

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
print(f"Confusion Matrix: \n{conf_matrix}")

# Compute other metrics with 95% CIs
for name, func in metrics.items():
    if name == "AUC-ROC":
        point_estimate = func(y_val, y_val_score[:, 1]) # only pass probabilities for the positive class
        lower, upper = bootstrap_ci(y_val, y_val_pred, y_val_score[:, 1], func) # only pass probabilities for the positive class
    else:
        point_estimate = func(y_val, y_val_pred)
        lower, upper = bootstrap_ci(y_val, y_val_pred, None, func)
    print(f"{name}: {point_estimate:.4f} [95% CI: {lower:.4f}, {upper:.4f}]")

    # define the parameter grid
param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# create an XGBoost classifier
clfxgb = XGBClassifier(random_state=13)

# initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(clfxgb, param_grid, cv=5, scoring='accuracy')

# train the classifier using the parameter grid
grid_search.fit(X_train_rfe, y_train)

# get the best hyperparameters and the best score achieved
print(f"Best hyperparameters:\n{grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# using optimal hyperparameters, evaluate performance on the validation set
clfxgb_best = grid_search.best_estimator_

# make predictions using the XGBoost model on validation data
y_val_score = clfxgb_best.predict_proba(X_val_rfe)[:, 1]
y_val_pred = clfxgb_best.predict(X_val_rfe)

# compute and print the confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
print(f"Confusion Matrix: \n{conf_matrix}")