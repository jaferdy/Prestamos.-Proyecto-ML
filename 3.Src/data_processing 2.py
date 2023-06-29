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


dict_Grade = df.groupby("Grade")["Loan Status"].mean()
dict_Grade = dict(dict_Grade)
df["Grade_ordered"] = df["Grade"].map(dict_Grade)

dict_SubGrade = df.groupby("Sub Grade")["Loan Status"].mean()
dict_SubGrade = dict(dict_SubGrade)
df["SubGrade_ordered"] = df["Sub Grade"].map(dict_SubGrade)

verification_status_ordered = df.groupby("Verification Status")["Loan Status"].mean()
verification_status_ordered = dict(verification_status_ordered)
df["verification_status_ordered"] = df["Verification Status"].map(verification_status_ordered)

home_ownership_ordered = df.groupby("Home Ownership")["Loan Status"].mean()
home_ownership_ordered = dict(home_ownership_ordered)
df["home_ownership_ordered"] = df["Home Ownership"].map(home_ownership_ordered)

initial_list_status_ordered = df.groupby("Initial List Status")["Loan Status"].mean()
initial_list_status_ordered = dict(initial_list_status_ordered)
df["initial_list_status_ordered"] = df["Initial List Status"].map(initial_list_status_ordered)

application_type_ordered = df.groupby("Application Type")["Loan Status"].mean()
application_type_ordered = dict(application_type_ordered)
df["application_type_ordered"] = df["Application Type"].map(application_type_ordered)

df.to_csv("./1.Data/2.Processed/processed2.csv")

print("Se ha terminado el procesamiento de datos.")
