from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, log_loss
import pickle
import pandas as pd
# from models import X_test, y_test

ruta_modelo = "./4.Models/trained_model1_UnderSampling.pkl"
with open(ruta_modelo, 'rb') as archivo_entrada:
        modelo = pickle.load(archivo_entrada)

test = pd.read_csv("./1.Data/4.Test/test.csv",index_col=0)
X = test.drop(["Loan Status"],axis=1)
y = test["Loan Status"]

y_pred = modelo.predict(X)
print("recall score", recall_score(y, y_pred))
print("precision_score", precision_score(y, y_pred))
print("accuracy_score", accuracy_score(y, y_pred))
print("f1_score", f1_score(y, y_pred))
print("log_loss", log_loss(y, y_pred))
print("log_loss", log_loss(y, y_pred))
print("confusion matrix",confusion_matrix(y, y_pred))
print("confusion matrix",confusion_matrix(y, y_pred,normalize="true"))

# y_pred = modelo.predict(X_test)
# print("recall score", recall_score(y_test, y_pred))
# print("precision_score", precision_score(y_test, y_pred))
# print("accuracy_score", accuracy_score(y_test, y_pred))
# print("f1_score", f1_score(y_test, y_pred))
# print("log_loss", log_loss(y_test, y_pred))
# print("log_loss", log_loss(y_test, y_pred))
# print("confusion matrix",confusion_matrix(y_test, y_pred))
# print("confusion matrix",confusion_matrix(y_test, y_pred,normalize="true"))