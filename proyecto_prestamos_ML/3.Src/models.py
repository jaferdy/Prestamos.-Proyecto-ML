import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, log_loss
import pickle
import os
import yaml

df = pd.read_csv("./1.Data/2.Processed/processed.csv")
X = df.drop(["Batch Enrolled","Loan Title","Loan Status"],axis=1)
y = df["Loan Status"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
train = pd.concat([X_train,y_train],axis=1)
test = pd.concat([X_test,y_test],axis=1)
train.to_csv("./1.Data/3.Train/train.csv")
test.to_csv("./1.Data/4.Test/test.csv")

under_sampler = RandomUnderSampler(random_state=42)
X_under, y_under = under_sampler.fit_resample(X_train,y_train)
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under,y_under,test_size=0.2,random_state=42)

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
rf_params = {
'scaler' : [StandardScaler(), None],
'classifier': [DecisionTreeClassifier(random_state=42)],
'classifier__max_features': [3,5,7],
'classifier__max_depth': [2,3,4,5]
}
knn_params = {
'scaler' : [StandardScaler(), None],
'classifier': [KNeighborsClassifier()],
'classifier__n_neighbors': [3,5,9]
}
gb_params = {
'scaler' : [StandardScaler(), None],
'classifier': [GradientBoostingClassifier(random_state=42)],
'classifier__max_features': [3,5,7],
'classifier__max_depth': [2,3,4,5]}

search_space = [rf_params, gb_params,knn_params]

modelo1_undersampling = GridSearchCV(estimator = pipe, param_grid = search_space, cv=5, scoring="f1", n_jobs=-1)
modelo1_undersampling.fit(X_train_under, y_train_under)

print(modelo1_undersampling.best_score_)
print(modelo1_undersampling.best_params_)
print(modelo1_undersampling.best_estimator_)

y_pred_under = modelo1_undersampling.best_estimator_.predict(X_test_under)
print("recall score", recall_score(y_test_under, y_pred_under))
print("precision_score", precision_score(y_test_under, y_pred_under))
print("accuracy_score", accuracy_score(y_test_under, y_pred_under))
print("f1_score", f1_score(y_test_under, y_pred_under))
print("log_loss", log_loss(y_test_under, y_pred_under))
print("log_loss", log_loss(y_test_under, y_pred_under))
print("confusion matrix",confusion_matrix(y_test_under, y_pred_under))
print("confusion matrix",confusion_matrix(y_test_under, y_pred_under,normalize="true"))


carpeta = "./4.Models"
nombre_archivo_pkl = 'trained_model1_UnderSampling.pkl'

# Crear la ruta completa
ruta_completa_pkl = os.path.join(carpeta, nombre_archivo_pkl)

with open(ruta_completa_pkl, 'wb') as archivo_salida:
    pickle.dump(modelo1_undersampling.best_estimator_, archivo_salida)

nombre_archivo_yaml = "model_config.yaml"
ruta_completa_yaml = os.path.join(carpeta, nombre_archivo_yaml)

with open(ruta_completa_yaml, "w") as f:
    yaml.dump(modelo1_undersampling.best_params_, f)
