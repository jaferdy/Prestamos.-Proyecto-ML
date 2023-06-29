# Prestamos.-Proyecto-ML

En este proyecto vamos a intentar predecir si en una serie de préstamos va a existir morosidad.

La fuente de los datos la hemos obtenido a través de Kaggle, pero a su vez vienen de la competición lanzada por Deloitte India. Abajo se muestra el enlace de la competición original.

Fuente: https://machinehack.com/hackathons/deloitte_presents_machine_learning_challenge_predict_loan_defaulters/overview

La estructura de carpetas es la siguiente:

1. Data. En ella disponemos de los datos en crudo en Raw, los datos procesados en Processed, los datos de train y los de test. Los tres últimos anteriores fueron creados a partir del script de python data_processing 3.py, que se encuentra en la carpeta 3.Src. En processed habrá 3 archivos, pero para el ejercicio usaremos processed3.csv. En la carpeta solo se muestra el processed3. Esto se debe a que el tamaño de processed2.csv y processed1.csv son superiores a 25 megas y no es posible subirlos. Si el usuario desea revisarlos, podrá ejecutar el script de Python data_processing.py y data_processing 2.py.
2. Notebooks. En esta carpeta se encuentran archivos Jupyter Notebook en los que se muestra un pequeño EDA, el procesamiento de datos y el entrenamiento y evaluación de los modelos. Existen varias carpetas dentro de Notebooks en las que se muestran hojas de trabajo con distintas pruebas y resultado de distintos modelos. En dichas hojas de trabajo, si queremos ejecutarlas, habrá que prestar atención a la ruta de la carpeta a la hora de leer los csv, ya que estos ficheros se han reorganizado una vez hemos optado por otros que se adaptan mejor a nuestro problema.
3. Src. En ella se encuentran distintos scripts de Python con el que se han creado distintos csv, yaml y pkl. 
4. Models. Se encuentran los mejores modelos que han sido entrenados y los que usaremos para predecir la variable target.
5. App. Streamlit.
6. Docs. Se encuentra la presentación de negocio.



