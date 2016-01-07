##### Librerias necesarias #####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_predict, cross_val_score

print "Metodo de Naive Bayes"

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


##### Crear el modelo de Naive Bayes #####
#Simple
modelS = GaussianNB()

#Cruzado
modelC = GaussianNB()


##### entrenamos la red	#####
modelS.fit(data_x_entre, data_y_entre)


##### Predicciones #####
#Simple
predic = modelS.predict(data_x_prueb)

#Cruzado
predic_cross = cross_val_predict(modelC,data_x,data_y,cv=10)


##### Score #####
#Simple
scor = modelS.score(data_x_prueb, data_y_prueb)

#Cruzado
scor_cross = cross_val_score(modelC,data_x, data_y,cv=10).mean()


##### Error cuadratico #####
#Simple
cuadr = np.mean((predic - data_y_prueb) ** 2)

#Cruzado
cuadr_cross = np.mean((predic_cross - data_y) ** 2)


##### Resumen de los ajustes del modelo #####
nombres = ['C. Bueno', 'C. Malo']
#Simple
print '\n******** Entrenamiento Simple ********'
#Score: 1 es la prediccion perfecta
print 'Score: %.2f'%scor
#Los errores cuadraticos
print "Suma residual de cuadrados: %.2f"%cuadr
print metrics.classification_report(data_y_prueb, predic, target_names=nombres)
conf = metrics.confusion_matrix(data_y_prueb, predic)
matr = {'Predic '+nombres[k]:conf[:,k] for k in range(len(nombres))}
print pd.DataFrame(data=matr,index=nombres)


#Cruzado
print '\n******** Entrenamiento 10-Cruzado ********'
#Score: 1 es la prediccion perfecta
print 'Score: %.2f'%scor_cross
#Los errores cuadraticos
print "Suma residual de cuadrados: %.2f"%cuadr_cross
print metrics.classification_report(data_y_prueb, predic_cross[-50:], target_names=nombres)
conf_cross = metrics.confusion_matrix(data_y_prueb, predic_cross[-50:])
matr_cross = {'Predic '+nombres[k]:conf_cross[:,k] for k in range(len(nombres))}
print pd.DataFrame(data=matr_cross,index=nombres)

a = str(predic)
b = str(predic_cross[-50:])
f = open('Nai_pred.txt','w')
f.write(a+'\n')
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
