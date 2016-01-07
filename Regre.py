##### Librerias necesarias #####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score, cross_val_predict

print "Metodo de Regresion Lineal"

##### Datos #####
#Importar german.data-numeric
data = np.loadtxt('german.data-numeric', delimiter=' ')

#Separar datos
data_x = data[:,:-1]
data_y = data[:,-1]

#Data de entrenamiento
data_x_entre = data_x[:-50]
data_y_entre = data_y[:-50]

#Data de prueba
data_x_prueb = data_x[-50:]
data_y_prueb = data_y[-50:]


##### Crear el model de Regresion Lineal #####
#Simple
modelS = LinearRegression()

#cruzado
modelC = LinearRegression()


##### Entrenamos del modelo #####
#Simple
modelS.fit(data_x_entre, data_y_entre)


##### Predicciones #####
#Simple
predic = modelS.predict(data_x_prueb)

#Cruzada
predic_cross = cross_val_predict(modelC, data_x, data_y, cv=10)


##### Score #####
#simple
scor = modelS.score(data_x_prueb, data_y_prueb)

#Cruzada
scor_cross = cross_val_score(modelC,data_x, data_y,cv=10).mean()


##### Error cuadratico #####
#Simple
cuadr = np.mean((predic - data_y_prueb) ** 2)

#Cruzado
cuadr_cross = np.mean((predic_cross - data_y) ** 2)


#####Tratamiento de los resultados #####
#simple
n = len(predic)
predic_resul = []
for i in range(n):
	if predic[i] <=1.5:
		predic_resul.append(1.)
	else:
		predic_resul.append(2.)
#cruzado
m = len(predic_cross)
predic_cross_resul = []
for j in range(m):
	if predic_cross[j] <=1.5:
		predic_cross_resul.append(1.)
	else:
		predic_cross_resul.append(2.)

##### Resumen de los ajustes del modelo #####
nombres = ['C. Bueno', 'C. Malo']
#Simple
#Los coeficientes
print '\n******** Entrenamiento Simple ********'
#print 'Coefficients: \n', modelS.coef_
#Score: 1 es la prediccion perfecta
print 'Score: %.2f'%scor
#Los errores cuadraticos
print "Suma residual de cuadrados: %.2f"%cuadr
print metrics.classification_report(data_y_prueb, predic_resul, target_names=nombres)
conf = metrics.confusion_matrix(data_y_prueb, predic_resul)
matr = {'Predic '+nombres[k]:list(conf[:,k]) for k in range(len(nombres))}
print pd.DataFrame(data=matr,index=nombres)

#Cruzado
#Los errores cuadraticos
print '\n******** Entrenamiento 10-Cruzado ********'
#Score: 1 es la prediccion perfecta
print 'Score: %.2f'%scor_cross
#Los errores cuadraticos
print "Suma residual de cuadrados: %.2f"%cuadr_cross
print metrics.classification_report(data_y_prueb, predic_cross_resul[-50:], target_names=nombres)
conf_cross = metrics.confusion_matrix(data_y_prueb, predic_cross_resul[-50:])
matr_cross = {'Predic '+nombres[k]:list(conf_cross[:,k]) for k in range(len(nombres))}
print pd.DataFrame(data=matr_cross,index=nombres)

##### Guardado #####
#convertir los valores en string
a = str(predic_resul)
b = str(predic_cross_resul[-50:])
#abrir el archivo para escribir
f = open('Reg_pred.txt','w')
#escribir de los valores predecidos simples
f.write(a+'\n')
#escribir de los valores predecidos cruzados
f.write(b+'\n')
f.close()

##### Ploteando #####
#Simple
fig1,ax = plt.subplots()
ax.plot([data_y_prueb.min(), data_y_prueb.max()], [data_y_prueb.min(), data_y_prueb.max()], 'r--', lw=4)
ax.scatter(data_y_prueb,predic,color='green')
ax.set_xlabel('Data Real')
ax.set_ylabel('Data Predecida')
ax.set_title('Prediccion Simple\nData real vs. Data predecida')

fig12,ax2 = plt.subplots()
ax2.plot([0, 50], [1.5, 1.5], 'r--', lw=4)
ax2.scatter(range(50),predic,color='green')
ax2.set_xlabel('Registros')
ax2.set_ylabel('Data Predecida')
ax2.set_title('Prediccion Simple\n Registro vs. Data Predecida')

#Cruzado
fig2,ay = plt.subplots()
ay.plot([data_y_prueb.min(), data_y_prueb.max()], [data_y_prueb.min(), data_y_prueb.max()], 'r--', lw=4)
ay.scatter(data_y_prueb,predic_cross[-50:])
ay.set_xlabel('Data Real')
ay.set_ylabel('Data Predecida')
ay.set_title('Prediccion Cruzada\nData real vs. Data predecida')

fig22,ay2 = plt.subplots()
ay2.plot([0, 50], [1.5, 1.5], 'r--', lw=4)
ay2.scatter(range(50),predic_cross[-50:])
ay2.set_xlabel('Registros')
ay2.set_ylabel('Data Predecida')
ay2.set_title('Prediccion Cruzada\n Registro vs. Data Predecida')

plt.show()
