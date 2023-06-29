import pandas as pd
import numpy as np

print("Comienza el procesamiento de datos.")

df = pd.read_csv("./1.Data/1.Raw/raw.csv",index_col=0)
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

df.drop(["Employment Duration","grado_experiencia","Debit to Income","grado_debit_income","Funded Amount",
         "Interest Rate","Inquires - six months","Open Account","Revolving Balance","Total Revolving Credit Limit",
         "Delinquency - two years","Total Accounts","Public Record","Collection 12 months Medical",
         "Total Received Interest","Total Received Late Fee","Loan Title","Batch Enrolled","Grado_interes",
         "Sub Grade","Grade",'Verification Status',"Home Ownership","Initial List Status","Application Type"],axis=1,
         inplace=True)

df.to_csv("./1.Data/2.Processed/processed3.csv")

print("Se ha terminado el procesamiento de datos.")
