# Prestamos.-Proyecto-ML

En este proyecto vamos a intentar predecir si en una serie de préstamos va a existir morosidad.

La fuente de los datos la hemos obtenido a través de Kaggle, pero a su vez vienen de la competición lanzada por Deloitte India. Abajo se muestra el enlace de la competición original.

Fuente: https://machinehack.com/hackathons/deloitte_presents_machine_learning_challenge_predict_loan_defaulters/overview

La estructura de carpetas es la siguiente:

1. Data. En ella disponemos de los datos en crudo en Raw, los datos procesados en Processed, los datos de train y los de test. Los tres últimos anteriores fueron creados a partir del script de python data_processing 3.py, que se encuentra en la carpeta 3.Src. En processed hay 3 archivos, pero para el ejercicio usaremos processed3.csv. 
2. Notebooks. En esta carpeta se encuentran archivos Jupyter Notebook en los que se muestra un pequeño EDA, el procesamiento de datos y el entrenamiento y evaluación de los modelos.
3. Src. En ella se encuentran distintos scripts de Python con el que se han creado distintos csv, yaml y pkl. Todos los documentos creados se encuentran a lo largo de las carpetas.
4. Models. Se encuentran los mejores modelos que han sido entrenados y los que usaremos para predecir la variable target.
5. App. Streamlit.
6. Docs. Se encuentra la presentación de negocio.



