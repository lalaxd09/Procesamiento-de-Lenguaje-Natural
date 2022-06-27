
import pandas as pd
from collections import OrderedDict
import numpy as np
import datetime
from Preprocesamiento.lematizador import lematizar

def pruebadedatos(X):
    lista_datos_t=[]
    lista_datos_o=[]

    for i in range(len(X)):
        if(str(type(X[i][1]))!="<class 'str'>"):
            lista_datos_o.append(str(type(X[i][1])))

        if(str(type(X[i][0]))!="<class 'str'>"):
            lista_datos_t.append(str(type(X[i][0])))

    print(list(OrderedDict.fromkeys(lista_datos_o)))
    print(list(OrderedDict.fromkeys(lista_datos_t)))
    print(len(X))

def guardarcorpus(lista_t_o,y_polarity,y_attraction):
    DocumentoSin={
        "titleOpinion":lista_t_o,
        "Polarity":y_polarity,
        'Attraction':y_attraction}

    df_rrss=pd.DataFrame(DocumentoSin)
    df_rrss.to_csv("SavedCorpus.csv")

def lem(n):
	cadena_lematizada = lematizar(n)
	#print (cadena_lematizada.lower())
	return cadena_lematizada

df = pd.read_excel("Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx")
indexNames = df[ df['Polarity'] == 3 ].index
df.drop(indexNames , inplace=True)
X = df.drop(['Polarity', 'Attraction'],axis=1).values
y_polarity = df['Polarity'].values
y_attraction = df['Attraction'].values	

lista_t_o=[]
for i in range(len(X)):

    if(str(type(X[i][0]))=="<class 'str'>" and str(type(X[i][1]))=="<class 'str'>" ):
        aux=X[i][0]+" "+X[i][1]
        lista_t_o.append(aux)
    elif(str(type(X[i][1]))!="<class 'str'>"):
        auxpre=str(X[i][1])
        aux=X[i][0]+" "+auxpre
        lista_t_o.append(aux)
    elif(str(type(X[i][0]))=="<class 'datetime.datetime'>"):
        fecha=X[i][0]
        fecha_str = fecha.strftime('%d/%m/%Y')
        aux=fecha_str+" "+X[i][1]
        lista_t_o.append(aux)
    else:
        auxpre=str(X[i][0])
        aux=auxpre+" "+X[i][1]
        lista_t_o.append(aux)

lista_t_o_lem=[]
for i in lista_t_o:
    lista_t_o_lem.append(lem(i))

guardarcorpus(lista_t_o_lem,y_polarity,y_attraction)
    



