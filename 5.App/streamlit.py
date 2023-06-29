import streamlit as st
import pandas as pd
from PIL import Image # !pip install Pillow
import streamlit.components.v1 as components
import plotly.express as px
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pickle

#Extrar los archivos pickle
with open('lin_reg.pkl', 'rb') as li: ### Escribir el nombre del pkl que vayamos a usar
    lin_reg = pickle.load(li)         ### Modificar el nombre del modelo

with open('log_reg.pkl', 'rb') as lo: ### Escribir el nombre del pkl que vayamos a usar
    log_reg = pickle.load(lo)         ### Modificar el nombre del modelo

with open('svc_m.pkl', 'rb') as sv:  ### Escribir el nombre del pkl que vayamos a usar
    svc_m = pickle.load(sv)          ### Modificar el nombre del modelo

#funcion para clasificar si habrá morosidad o no en el préstamo. 
def classify(num):
    if num == 0:
        return 'No hay morosidad'
    elif num == 1:
        return 'Morosidad'

def main():
    #titulo
    st.title("Predicción Morosidad en préstamos bancarios")
    #titulo de sidebar
    st.sidebar.header('User Input Parameters')

    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():
        Loan_Amount = st.sidebar.slider('Loan Amount', 1014.0, 35000,0, 16848.9)
        Funded_Amount = st.sidebar.slider('Funded_Amount', 1014.0, 35000,0, 15770.6)
        Funded_Amount_Investor = st.sidebar.slider('Funded_Amount_Investor', 1.0, 6.9, 1.3)
        Term = st.sidebar.slider('Term', 36, 59, 48)
        Interest_Rate = st.sidebar.slider('Interest_Rate', 5.3, 27.2, 11.8)
        Employment_Duration = st.sidebar.slider('Employment_Duration', 1.5, 46.5, 10.0)
        Debit_to_Income = st.sidebar.slider('Debit_to_Income', 0.67, 39.6, 23.3)
        #### A continuación, faltan por meter los datos mínimos, máximos y por defecto de cada categoría.
        Revolving_Balance = st.sidebar.slider('Revolving_Balance', 0.67, 39.6, 23.3)
        Revolving_Utilities = st.sidebar.slider('Revolving_Utilities', 0.67, 39.6, 23.3)
        Total_Received_Interest = st.sidebar.slider('Total_Received_Interest', 0.67, 39.6, 23.3)
        Total_Received_Late_Fee = st.sidebar.slider('Total_Received_Late_Fee', 0.67, 39.6, 23.3)
        Recoveries = st.sidebar.slider('Recoveries', 0.67, 39.6, 23.3)
        Collection_Recovery_Fee = st.sidebar.slider('Collection_Recovery_Fee', 0.67, 39.6, 23.3)
        Last_week_Pay = st.sidebar.slider('Last_week_Pay', 0.67, 39.6, 23.3)
        Total_Collection_Amount = st.sidebar.slider('Total_Collection_Amount', 0.67, 39.6, 23.3)
        Total_Current_Balance = st.sidebar.slider('Total_Current_Balance', 0.67, 39.6, 23.3)
        Total_Revolving_Credit_Limit = st.sidebar.slider('Total_Revolving_Credit_Limit', 0.67, 39.6, 23.3)
        Proporción_Morosidad_según_financiación = st.sidebar.slider('Proporción_Morosidad_según_financiación', 0.67, 39.6, 23.3)
        Proporción_Morosidad_segun_Batch_Enrolled = st.sidebar.slider('Proporción_Morosidad_segun_Batch_Enrolled', 0.67, 39.6, 23.3)
        Grado_interes_ordered = st.sidebar.slider('Grado_interes_ordered', 0.67, 39.6, 23.3)
        Grade_ordered = st.sidebar.slider('Grade_ordered', 0.67, 39.6, 23.3)
        SubGrade_ordered = st.sidebar.slider('SubGrade_ordered', 0.67, 39.6, 23.3)
        application_type_ordered = st.sidebar.slider('application_type_ordered', 0.67, 39.6, 23.3)
        
        data = {'Loan_Amount': Loan_Amount,
                'Funded_Amount': Funded_Amount,
                'Funded_Amount_Investor': Funded_Amount_Investor,
                'Term': Term,
                "Interest_Rate" : Interest_Rate,
                "Employment_Duration" : Employment_Duration,
                "Debit_to_Income" : Debit_to_Income,
                "Revolving_Balance" : Revolving_Balance,
                "Revolving_Utilities" : Revolving_Utilities,
                "Total_Received_Interest" : Total_Received_Interest,
                "Total_Received_Late_Fee" : Total_Received_Late_Fee,
                "Recoveries" : Recoveries,
                "Collection_Recovery_Fee" : Collection_Recovery_Fee,
                "Last_week_Pay" : Last_week_Pay,
                "Total_Collection_Amount" : Total_Collection_Amount,
                "Total_Current_Balance" : Total_Current_Balance,
                "Total_Revolving_Credit_Limit" : Total_Revolving_Credit_Limit,
                "Proporción_Morosidad_según_financiación" : Proporción_Morosidad_según_financiación,
                "Proporción_Morosidad_segun_Batch_Enrolled" : Proporción_Morosidad_segun_Batch_Enrolled,
                "Grado_interes_ordered" : Grado_interes_ordered,
                "Grade_ordered" : Grade_ordered,
                "SubGrade_ordered" : SubGrade_ordered,
                "application_type_ordered" : application_type_ordered
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    #escoger el modelo preferido
    option = ['Linear Regression', 'Logistic Regression', 'SVM']
    model = st.sidebar.selectbox('Which model you like to use?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_m.predict(df)))

if __name__ == '__main__':
    main()