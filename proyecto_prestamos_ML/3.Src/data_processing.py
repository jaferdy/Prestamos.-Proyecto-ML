import pandas as pd
import numpy as np

df = pd.read_csv("./1.Data/1.Raw/raw.csv",index_col=0)
mapeo_columnas = {"Employment Duration":"Home Ownership",
                "Home Ownership":"Employment Duration"}
df = df.rename(columns=mapeo_columnas) # Cambiamos nombres de columnas. En el fichero original están con los nombres intercambiados.
df.drop("Accounts Delinquent",axis=1,inplace=True) # Es un valor constante.
sub_grade_ordered = {"G5":0,"G4":1,"G3":2,"G2":3,"G1":4,"F5":5,"F4":6,"F3":7,"F2":8,"F1":9,"E5":10,"E4":11,"E3":12,"E2":13,
"E1":14,"D5":15,"D4":16,"D3":17,"D2":18,"D1":19,"C5":20,"C4":21,"C3":22,"C2":23,"C1":24,"B5":25,"B4":26,"B3":27,"B2":28,
"B1":29,"A5":30,"A4":31,"A3":32,"A2":33,"A1":34} # Asignamos valor a dichos valores de menor a mayor. En economía, el mayor valor crediticio se corresponde a la letra A.
df["Sub Grade_ordered"] = df["Sub Grade"].map(sub_grade_ordered) # Mapeamos valores anteriores
df.drop("Sub Grade",axis=1,inplace=True) 
grade_ordered = {"G":0,"F":1,"E":2,"D":3,"C":4,"B":5,"A":6} # Hacemos lo mismo que en sub grade por la misma razón.
df["Grade_ordered"] = df["Grade"].map(grade_ordered)
df.drop("Grade",axis=1,inplace=True)
verification_status_ordered = {"Not Verified":0,
                               "Verified":1,
                               "Source Verified": 2}
df["verification_status_ordered"] = df['Verification Status'].map(verification_status_ordered)
df.drop('Verification Status',axis=1,inplace=True)
df.drop('Payment Plan',axis=1,inplace=True) # Es un valor constante.
home_ownership_ordered = {"RENT":0,
                          "MORTGAGE":1,
                          "OWN":2} 
df["Home Ownership ordered"] = df["Home Ownership"].map(home_ownership_ordered)
df.drop("Home Ownership",axis=1,inplace=True)
initial_list_status_ordered = {"w":0,
                               "f":1}
df["initial_list_status_ordered"] = df["Initial List Status"].map(initial_list_status_ordered)
df.drop("Initial List Status",axis=1,inplace=True)
application_type_ordered = {"INDIVIDUAL":0,
                            "JOINT":1}
df["application_type_ordered"] = df["Application Type"].map(application_type_ordered)
df.drop("Application Type",axis=1, inplace=True)

df.to_csv("./1.Data/2.Processed/processed.csv")

