import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,\
                            roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,log_loss, get_scorer_names
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
import yaml
import os

df = pd.read_csv("./1.Data/2.Processed/processed3.csv")
X = df.drop(["Loan Status",'Funded Amount/Interest Rate',"Term"],axis=1)
y = df["Loan Status"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
train = pd.concat([X_train,y_train],axis=1)
test = pd.concat([X_test,y_test],axis=1)
train.to_csv("./1.Data/3.Train/train.csv")
test.to_csv("./1.Data/4.Test/test.csv")

under_sampler = RandomUnderSampler(random_state=42)
X_under, y_under = under_sampler.fit_resample(X_train,y_train)
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under,y_under,test_size=0.2,random_state=42)

# Se indican abajo los modelos que mejores resultados nos han dado. 
# Los modelos probados se encuentran en:
# 2.Notebooks\2a. Entrenamiento_evaluacion_Modelos f1 P3.ipynb
# 2.Notebooks\2b. Entrenamiento_evaluacion_Modelos recall P3.ipynb

# Model scoring f1.

piperf = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("selectkbest", SelectKBest()),
    ("classifier", RandomForestClassifier())
])
rf_params = {
'scaler' : [StandardScaler(), None],
'selectkbest__k' : np.arange(5,15),
'classifier__max_features': [3,5,6,7],
'classifier__max_depth': [2,3,4,5,6]
}
modelrf = GridSearchCV(estimator = piperf, param_grid = rf_params, cv=3, scoring="f1",n_jobs=-1)

modelrf.fit(X_train_under, y_train_under)
print("modelrf best score:",modelrf.best_score_ )
print("modelrf best params:",modelrf.best_params_ )
print("modelrf best estimator:",modelrf.best_estimator_ )
y_pred_under_rf = modelrf.best_estimator_.predict(X_test_under)
print("modelrf train recall score", recall_score(y_test_under, y_pred_under_rf))
print("modelrf train precision_score", precision_score(y_test_under, y_pred_under_rf))
print("modelrf train accuracy_score", accuracy_score(y_test_under, y_pred_under_rf))
print("modelrf train f1_score", f1_score(y_test_under, y_pred_under_rf))
print("modelrf train log_loss", log_loss(y_test_under, y_pred_under_rf))
print("modelrf train confusion matrix",confusion_matrix(y_test_under, y_pred_under_rf))
print("modelrf train confusion matrix",confusion_matrix(y_test_under, y_pred_under_rf,normalize="true"))


# Model scoring recall.

piperf2 = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("selectkbest", SelectKBest()),
    ("classifier", RandomForestClassifier())
])
rf_params2 = {
'scaler' : [StandardScaler(), None],
'selectkbest__k' : np.arange(5,15),
'classifier__max_features': [3,5,6,7],
'classifier__max_depth': [2,3,4,5,6]
}
modelrf2 = GridSearchCV(estimator = piperf2, param_grid = rf_params2, cv=3, scoring="recall")

modelrf2.fit(X_train_under, y_train_under)
print("modelrf best score:",modelrf2.best_score_ )
print("modelrf best params:",modelrf2.best_params_ )
print("modelrf best estimator:",modelrf2.best_estimator_ )
y_pred_under_rf2 = modelrf2.best_estimator_.predict(X_test_under)
print("modelrf train recall score", recall_score(y_test_under, y_pred_under_rf2))
print("modelrf train precision_score", precision_score(y_test_under, y_pred_under_rf2))
print("modelrf train accuracy_score", accuracy_score(y_test_under, y_pred_under_rf2))
print("modelrf train f1_score", f1_score(y_test_under, y_pred_under_rf2))
print("modelrf train log_loss", log_loss(y_test_under, y_pred_under_rf2))
print("modelrf train confusion matrix",confusion_matrix(y_test_under, y_pred_under_rf2))
print("modelrf train confusion matrix",confusion_matrix(y_test_under, y_pred_under_rf2,normalize="true"))

carpeta1 = "./4.Models"
nombre_archivo_pkl1 = 'ModeloRF f1 p3.pkl'

# Crear la ruta completa
ruta_completa_pkl1 = os.path.join(carpeta1, nombre_archivo_pkl1)

with open(ruta_completa_pkl1, 'wb') as archivo_salida:
    pickle.dump(modelrf.best_estimator_, archivo_salida)

nombre_archivo_yaml1 = "ModeloRF f1 p3.yaml"
ruta_completa_yaml1 = os.path.join(carpeta1, nombre_archivo_yaml1)

with open(ruta_completa_yaml1, "w") as f:
    yaml.dump(modelrf.best_params_, f)

carpeta2 = "./4.Models"
nombre_archivo_pkl2 = 'ModeloRF recall p3.pkl'

# Crear la ruta completa
ruta_completa_pkl2 = os.path.join(carpeta2, nombre_archivo_pkl2)

with open(ruta_completa_pkl2, 'wb') as archivo_salida:
    pickle.dump(modelrf2.best_estimator_, archivo_salida)

nombre_archivo_yaml2 = "ModeloRF recall p3.yaml"
ruta_completa_yaml2 = os.path.join(carpeta1, nombre_archivo_yaml2)

with open(ruta_completa_yaml2, "w") as f:
    yaml.dump(modelrf2.best_params_, f)
