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
with open('../4.Models/ModeloRF_f1_p3.pkl', 'rb') as li: 
    modelRF_f1 = pickle.load(li)        

with open('../4.Models/ModeloRF_recall_p3.pkl', 'rb') as lo: 
    modelRF_recall = pickle.load(lo)

with open('../4.Models/ModeloGB recall.pkl', 'rb') as li: 
    modelGB_recall = pickle.load(li)

with open('../4.Models/Modelxgb_roc_auc.pkl', 'rb') as sv: 
    modelxgb_roc = pickle.load(sv)

with open('../4.Models/ModelAB f1.pkl', 'rb') as li: 
    modelRF_ab_f1 = pickle.load(li)

         
df = pd.read_csv("../1.Data/1.Raw/raw.csv",index_col=0)
mapeo_columnas = {"Employment Duration":"Home Ownership",
                "Home Ownership":"Employment Duration"}
df.rename(columns=mapeo_columnas,inplace=True)

df.drop(["Accounts Delinquent","Payment Plan"],axis=1,inplace=True)

df["Employment Duration"]=df["Employment Duration"]/(365.25 * 24)
grado_experiencia = []
for i in range(len(df)):
    if df["Employment Duration"].iloc[i] <= 5.896628260894365:
        grado_experiencia.append("Experiencia baja")
    elif df["Employment Duration"].iloc[i] > 5.896628260894365 and df["Employment Duration"].iloc[i] <= 7.9096318366415685:
        grado_experiencia.append("Experiencia media baja")
    elif df["Employment Duration"].iloc[i] > 7.9096318366415685 and df["Employment Duration"].iloc[i] <= 10.79435578199863:
        grado_experiencia.append("Experiencia media")
    elif df["Employment Duration"].iloc[i] > 10.79435578199863 and df["Employment Duration"].iloc[i] <= 13.243219542550763:
        grado_experiencia.append("Experiencia alta")
    else:
        grado_experiencia.append("Experiencia muy alta")
df["grado_experiencia"]= grado_experiencia
grado_experiencia = {"Experiencia baja":0,
                     "Experiencia media baja":1,
                     "Experiencia media": 3,
                     "Experiencia alta": 6,
                     "Experiencia muy alta": 9}
df["grado_experiencia_ordered"] = df["grado_experiencia"].map(grado_experiencia)

grado_debit_income = []
for i in range(len(df)):
    if df["Debit to Income"].iloc[i] <= 18.512048193200002:
        grado_debit_income.append("Debit income bueno")
    elif df["Debit to Income"].iloc[i] > 18.512048193200002 and df["Debit to Income"].iloc[i] <= 27.591865592999998:
        grado_debit_income.append("Debit income medio")
    else:
        grado_debit_income.append("Debit income malo")
df["grado_debit_income"]= grado_debit_income
grado_debit_income = {"Debit income malo":0,
                     "Debit income medio":1.5,
                     "Debit income bueno": 3}
df["grado_debit_income_ordered"] = df["grado_debit_income"].map(grado_debit_income)

df["Funded Amount/Interest Rate"]=df["Funded Amount"]/df["Interest Rate"]
df["Funded Amount / Debit to Income"]=df["Funded Amount"]/df["Debit to Income"]
df["Inquires - six months / Open Account"] = df["Inquires - six months"]/df["Open Account"]
df["Revolving Balance / Total Revolving Credit Limit"]=df["Revolving Balance"]/df["Total Revolving Credit Limit"]
df["Delinquency - two years / Total Accounts"]=df["Delinquency - two years"]/df["Total Accounts"]
df["cuentas públicas prestatario"] =df["Public Record"]+df["Collection 12 months Medical"]
df["Eficacia pagos respecto financiación inicial"] =(df["Total Received Interest"]+df["Total Received Late Fee"])/df["Funded Amount"]

dict_Loan_Title = df.groupby("Loan Title")["Loan Status"].mean()
dict_Loan_Title = dict(dict_Loan_Title)
df["Proporción Morosidad según financiación"] = df["Loan Title"].map(dict_Loan_Title)

dict_batch_enrolled = df.groupby("Batch Enrolled")["Loan Status"].mean()
dict_batch_enrolled = dict(dict_batch_enrolled)
df["Proporción Morosidad según Batch Enrolled"] = df["Batch Enrolled"].map(dict_batch_enrolled)

grado_interes = []
for i in range(len(df)):
    if df["Interest Rate"].iloc[i] <= 9.2971471585:
        grado_interes.append("Interes bajo")
    elif df["Interest Rate"].iloc[i] > 9.2971471585 and df["Interest Rate"].iloc[i] <= 11.37769635:
        grado_interes.append("Interes medio bajo")
    elif df["Interest Rate"].iloc[i] > 11.37769635 and df["Interest Rate"].iloc[i] <= 14.193533065:
        grado_interes.append("Interes medio alto")
    elif df["Interest Rate"].iloc[i] > 14.193533065 and df["Interest Rate"].iloc[i] <= 16.64172601825:
        grado_interes.append("Interes alto")
    else:
        grado_interes.append("Interes muy alto")
df["Grado_interes"]=grado_interes
grado_interes_ordered = {"Interes muy alto":0,
                               "Interes alto":1,
                               "Interes medio alto": 3,
                               "Interes medio bajo": 5,
                               "Interes bajo": 7}
df["grado_interes_ordered"] = df["Grado_interes"].map(grado_interes_ordered)

sub_grade_ordered = {"G5":0,"G4":1,"G3":2,"G2":3,"G1":4,
                     "F5":8,"F4":9,"F3":10,"F2":11,"F1":12,
                     "E5":16,"E4":17,"E3":18,"E2":19,"E1":20,
                     "D5":25,"D4":26,"D3":27,"D2":28,"D1":29,
                     "C5":35,"C4":36,"C3":37,"C2":38,"C1":39,
                     "B5":45,"B4":46,"B3":47,"B2":48,"B1":49,
                     "A5":55,"A4":56,"A3":57,"A2":58,"A1":59}
#Asignamos valor a dichos valores de menor a mayor. En economía, el mayor valor crediticio se corresponde a la letra A.
df["Sub Grade_ordered"] = df["Sub Grade"].map(sub_grade_ordered) # Mapeamos valores anteriores

grade_ordered = {"G":0,"F":2,"E":5,"D":8,"C":12,"B":16,"A":20} # Hacemos lo mismo que en sub grade por la misma razón.
df["Grade_ordered"] = df["Grade"].map(grade_ordered)

verification_status_ordered = {"Not Verified":0,
                               "Verified":2,
                               "Source Verified": 4}
df["verification_status_ordered"] = df['Verification Status'].map(verification_status_ordered)

home_ownership_ordered = {"RENT":1,
                          "MORTGAGE":2,
                          "OWN":5} 
df["Home Ownership ordered"] = df["Home Ownership"].map(home_ownership_ordered)

initial_list_status_ordered = df.groupby("Initial List Status")["Loan Status"].mean()
initial_list_status_ordered = dict(initial_list_status_ordered)
df["initial_list_status_ordered"] = df["Initial List Status"].map(initial_list_status_ordered)

application_type_ordered = df.groupby("Application Type")["Loan Status"].mean()
application_type_ordered = dict(application_type_ordered)
df["application_type_ordered"] = df["Application Type"].map(application_type_ordered)


menu = st.sidebar.selectbox("Procesos", ['inicio','Análisis de datos (EDA)', 'Procesamiento de datos',"Modelos"])

if menu == "inicio":
    st.markdown("Bienvenido.\n\nA lo largo de aplicación, podremos ver cómo se ha resulto el problema de predecir si un préstamo bancario acabará teniendo morosidad.")
    st.markdown("Esta aplicación está compuesta por 3 pestañas, sin contar con esta inicial.\n\nAnálisis de datos (EDA)\n\nProcesamiento de datos\n\nModelos")

if menu == "Análisis de datos (EDA)":
    tab1, tab2, tab3, tab4, tab5,tab6,tab7 = st.tabs(['¿Problema Balanceado?','Mapa de correlaciones',
                                             "Outliers experiencia laboral", "Outliers tipo de interés",
                                               "Proporción morosidad en Batch Enrolled","Proporción morosidad sobre vivienda",
                                               "proporción de morosidad sobre el tipo de préstamo"])
    with tab1:
        fig = sns.displot(df["Loan Status"],color="red")
        st.pyplot(fig)
        st.markdown("Podemos observar que el problema está desbalanceado.")

    with tab2:
        plt.figure(figsize=(30,15))
        sns.heatmap(df.corr(numeric_only=True), annot=True)
        fig2 = plt.gcf()
        st.pyplot(fig2)
        st.markdown("Podemos observar que no existe correlación entre las variables.")
    # with tab3:
    #     fig3 = sns.boxplot(data=df,x="Employment Duration",orient="h")
    #     st.pyplot(fig3)
    #     st.markdown("Podemos observar que existen varios outliers.")
    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="Employment Duration", orient="h", ax=ax3)
        plt.xticks(rotation=90)
        st.pyplot(fig3)
        st.markdown("Podemos observar que existen varios outliers.")
      

    # with tab4:
    #     fig4 = sns.boxplot(data=df,x="Interest Rate",orient="h")
    #     st.pyplot(fig4)
    #     st.markdown("Podemos observar que existen outliers.")
    # with tab5:
    #     fig5= sns.countplot(data=df, x="Batch Enrolled", hue="Loan Status")
    #     st.pyplot(fig5)
    #     st.markdown("Observamos la proporción de morosidad sobre Batch Enrolled")
    # with tab6:
    #     fig6= sns.countplot(data=df, x="Home Ownership", hue="Loan Status")
    #     st.pyplot(fig6)
    #     st.markdown("Observamos la proporción de morosidad sobre Batch Enrolled")
    # with tab7:
    #     fig7 = sns.countplot(data=df, x="Loan Title", hue="Loan Status")
    #     st.pyplot(fig7)
    #     st.markdown("Observamos la proporción de morosidad sobre el tipo de préstamo")
    with tab4:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="Interest Rate", orient="h", ax=ax4)
        plt.xticks(rotation=90)
        st.pyplot(fig4)
        st.markdown("Podemos observar que existen outliers.")

    with tab5:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x="Batch Enrolled", hue="Loan Status", ax=ax5)
        plt.xticks(rotation=90)
        st.pyplot(fig5)
        st.markdown("Observamos la proporción de morosidad sobre Batch Enrolled")

    with tab6:
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x="Home Ownership", hue="Loan Status", ax=ax6)
        st.pyplot(fig6)
        st.markdown("Observamos la proporción de morosidad sobre Home Ownership")

    with tab7:
        fig7, ax7 = plt.subplots(figsize=(30, 10))
        sns.countplot(data=df, x="Loan Title", hue="Loan Status", ax=ax7)
        plt.xticks(rotation=90)
        st.pyplot(fig7)
        st.markdown("Observamos la proporción de morosidad sobre el tipo de préstamo")

if menu == 'Procesamiento de datos':
    tab8, tab9,tab10 = st.tabs(["Proporción morosidad según grado interés","Proporción morosidad según grado experiencia",
                                "Proporción morosidad según grado calificación"
                                ])
    with tab8:
        plt.figure(figsize=(30,15))
        fig8 = sns.countplot(x=df["Grado_interes"], hue=df["Loan Status"])
        st.pyplot(fig8.figure)
        st.markdown("Observamos la proporción de morosidad sobre el grado de tipo de interés")
    with tab9:
        plt.figure(figsize=(30,15))
        fig9 = sns.countplot(data=df,x="grado_experiencia",hue="Loan Status")
        st.pyplot(fig9.figure)
        st.markdown("Observamos la proporción de morosidad sobre el grado de experiencia")
    with tab10:
        fig10 = sns.countplot(data=df,x="Grade_ordered",hue="Loan Status")
        st.pyplot(fig9.figure)
        st.markdown("Observamos la proporción de morosidad sobre el grado de calificación")

def classify(num):
    if num == 0:
        return 'No hay morosidad'
    elif num == 1:
        return 'Morosidad'



    #funcion para poner los parametrso en el sidebar
def user_input_parameters():
    Loan_Amount = st.sidebar.slider('Loan Amount', 1014.0, 35000.0, 16848.9)
    Funded_Amount_Investor = st.sidebar.slider('Funded_Amount_Investor', 1.0, 6.9, 1.3)
    #### A continuación, faltan por meter los datos mínimos, máximos y por defecto de cada categoría.
    Revolving_Utilities = st.sidebar.slider('Revolving_Utilities', 1114.590204, 34999.746430)
    Recoveries = st.sidebar.slider('Recoveries', 0.000036, 4339.261318, 3.351983)
    Collection_Recovery_Fee = st.sidebar.slider('Collection_Recovery_Fee', 0.000045, 166.833000, 0.781117)
    Last_week_Pay = st.sidebar.slider('Last_week_Pay', 0.0, 161.0, 71.205707)
    Total_Collection_Amount = st.sidebar.slider('Total_Collection_Amount', 1.0, 16421.0, 146.513989)
    Total_Current_Balance = st.sidebar.slider('Total_Current_Balance', 6.170000e+02, 1.177412e+06, 1.591848e+05)
    grado_experiencia_ordered = st.sidebar.slider('grado_experiencia_ordered', 0, 9, 3)
    grado_debit_income_ordered = st.sidebar.slider('grado_debit_income_ordered', 0.0, 3.0, 1.5)
    Funded_Amount_Debit_to_Income = st.sidebar.slider('Funded Amount / Debit to Income', 34.616382,27329.355315, 807.705789)
    Inquires_six_months_Open_Account = st.sidebar.slider('Funded Amount / Debit to Income', 0,1, 0)
    Revolving_Balance_Total_Revolving_Credit_Limit = st.sidebar.slider('Revolving Balance / Total Revolving Credit Limit', 0.0,58.966197,0.3)
    Delinquency_two_years_Total_Accounts = st.sidebar.slider('Delinquency - two years / Total Accounts', 0,2,0)
    cuentas_publicas_prestatario = st.sidebar.slider('cuentas públicas prestatario', 0.0,5.0,0.0)
    Eficacia_pagos_respecto_financiación_inicial = st.sidebar.slider('Eficacia pagos respecto financiación inicial', 0.000164,8.898065,0.173464)
    Proporción_Morosidad_según_financiación = st.sidebar.slider('Proporción Morosidad según financiación', 0.0,0.5,0.092514)
    Proporción_Morosidad_segun_Batch_Enrolled = st.sidebar.slider('Proporción_Morosidad_segun_Batch_Enrolled', 0.078297, 0.125000, 0.092527)
    Grado_interes_ordered = st.sidebar.slider('Grado_interes_ordered', 0, 7, 5)
    Grade_ordered = st.sidebar.slider('Grade_ordered', 0, 20, 12)
    SubGrade_ordered = st.sidebar.slider('SubGrade_ordered', 0, 59, 38)
    verification_status_ordered = st.sidebar.slider('verification_status_ordered', 0, 4, 2)
    Home_Ownership_ordered = st.sidebar.slider('Home_Ownership_ordered', 1, 5, 2)
    initial_list_status_ordered = st.sidebar.slider('initial_list_status_ordered', 0.088901, 0.096714, 0.092523)
    application_type_ordered = st.sidebar.slider('application_type_ordered', 0.089431, 0.092516, 0.092510)

    data = {'Loan_Amount': Loan_Amount,
            'Funded_Amount_Investor': Funded_Amount_Investor,
            "Revolving_Utilities" : Revolving_Utilities,
            "Recoveries" : Recoveries,
            "Collection_Recovery_Fee" : Collection_Recovery_Fee,
            "Last_week_Pay" : Last_week_Pay,
            "Total_Collection_Amount" : Total_Collection_Amount,
            "Total_Current_Balance" : Total_Current_Balance,
            "grado_experiencia_ordered" : grado_experiencia_ordered,
            "grado_debit_income_ordered" : grado_debit_income_ordered,
            "Funded_Amount / Debit_to_Income" : Funded_Amount_Debit_to_Income,
            "Inquires - six months / Open Account": Inquires_six_months_Open_Account,
            "Revolving Balance / Total Revolving Credit Limit" : Revolving_Balance_Total_Revolving_Credit_Limit,
            "Delinquency_two years_Total_Accounts" : Delinquency_two_years_Total_Accounts ,
            "cuentas públicas prestatario" : cuentas_publicas_prestatario,
            "Eficacia pagos respecto financiación inicial" : Eficacia_pagos_respecto_financiación_inicial,
            "Proporción Morosidad según financiación" : Proporción_Morosidad_según_financiación,
            "Proporción_Morosidad_segun_Batch_Enrolled" : Proporción_Morosidad_segun_Batch_Enrolled,
            "Grado_interes_ordered" : Grado_interes_ordered,
            "Grade_ordered" : Grade_ordered,
            "SubGrade_ordered" : SubGrade_ordered,
            "verification_status_ordered" : verification_status_ordered,
            "Home_Ownership_ordered" : Home_Ownership_ordered,
            "initial_list_status_ordered" : initial_list_status_ordered,
            "application_type_ordered" : application_type_ordered
            }
    features = pd.DataFrame(data, index=[0])
    return features

if menu == "Modelos":
    st.text('Modelo de predicción con mejores resultados:\nRandom forest con métrica f1')
    df = user_input_parameters()
    #escoger el modelo preferido
    option = ['modelRF_f1', 'modelRF_recall', 'modelGB_recall',"modelxgb_roc","modelRF_ab_f1"]
    model = st.sidebar.selectbox('Which model you like to use?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'modelRF_f1':
            st.success(classify(modelRF_f1.predict(df)))
        elif model == 'modelRF_recall':
            st.success(classify(modelRF_recall.predict(df)))
        elif model == 'modelGB_recall':
            st.success(classify(modelGB_recall.predict(df)))
        elif model == 'modelxgb_roc':
            st.success(classify(modelxgb_roc.predict(df)))
        else:
            st.success(classify(modelRF_ab_f1.predict(df)))
