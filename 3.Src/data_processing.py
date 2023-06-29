import pandas as pd
import numpy as np

print("Comienza el procesamiento de datos.")

df = pd.read_csv("./1.Data/1.Raw/raw.csv",index_col=0)
mapeo_columnas = {"Employment Duration":"Home Ownership",
                "Home Ownership":"Employment Duration"}
df.rename(columns=mapeo_columnas,inplace=True)

df.drop(["Accounts Delinquent","Payment Plan"],axis=1,inplace=True)

df["Employment Duration"]=df["Employment Duration"]/(365.25 * 24)
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
    if df["Interest Rate"].iloc[i]<=np.percentile(df["Interest Rate"],33):
        grado_interes.append("Interes bajo")
    elif df["Interest Rate"].iloc[i]>np.percentile(df["Interest Rate"],33) and df["Interest Rate"].iloc[i]<=np.percentile(df["Interest Rate"],67):
        grado_interes.append("Interes medio")
    else:
        grado_interes.append("Interes alto")
df["Grado_interes"]=grado_interes
Grado_interes_ordered=df.groupby("Grado_interes")["Loan Status"].mean()
Grado_interes_ordered=dict(Grado_interes_ordered)
df["Grado_interes_ordered"]=df["Grado_interes"].map(Grado_interes_ordered)

sub_grade_ordered = {"G5":0,"G4":1,"G3":2,"G2":3,"G1":4,"F5":5,
                     "F4":6,"F3":7,"F2":8,"F1":9,"E5":10,"E4":11,
                     "E3":12,"E2":13,"E1":14,"D5":15,"D4":16,"D3":17,
                     "D2":18,"D1":19,"C5":20,"C4":21,"C3":22,"C2":23,
                     "C1":24,"B5":25,"B4":26,"B3":27,"B2":28,"B1":29,
                     "A5":30,"A4":31,"A3":32,"A2":33,"A1":34}
#Asignamos valor a dichos valores de menor a mayor. En economía, el mayor valor crediticio se corresponde a la letra A.

df["Sub Grade_ordered"] = df["Sub Grade"].map(sub_grade_ordered) # Mapeamos valores anteriores
grade_ordered = {"G":0,"F":1,"E":2,"D":3,"C":4,"B":5,"A":6} # Hacemos lo mismo que en sub grade por la misma razón.
df["Grade_ordered"] = df["Grade"].map(grade_ordered)

verification_status_ordered = {"Not Verified":0,
                               "Verified":1,
                               "Source Verified": 2}
df["verification_status_ordered"] = df['Verification Status'].map(verification_status_ordered)

home_ownership_ordered = {"RENT":1,
                          "MORTGAGE":2,
                          "OWN":3} 
df["Home Ownership ordered"] = df["Home Ownership"].map(home_ownership_ordered)

initial_list_status_ordered = {"w":0,
                               "f":1}
df["initial_list_status_ordered"] = df["Initial List Status"].map(initial_list_status_ordered)

application_type_ordered = {"INDIVIDUAL":1,
                            "JOINT":2}
df["application_type_ordered"] = df["Application Type"].map(application_type_ordered)



df.to_csv("./1.Data/2.Processed/processed1.csv")


print("Se ha terminado el procesamiento de datos.")
