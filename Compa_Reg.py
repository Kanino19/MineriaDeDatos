##### Librerias necesarias #####
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score, cross_val_predict

print "Metodo de Regresion Lineal para Comparacion"


##### Datos #####
# Columnas de las variables a comparar
colx = 1
coly = 3

# Importar german.data-numeric
data = np.loadtxt('german.data-numeric', delimiter=' ')

# Separar datos
data_x = data[:,colx]
data_xT = np.zeros((len(data_x),1))
for i in range(len(data_x)):
	data_xT[i,0] = data_x[i]
data_y = data[:,coly]

# Data de entrenamiento
data_x_entre = data_xT[:-50]
data_y_entre = data_y[:-50]

# Data de prueba
data_x_prueb = data_xT[-50:]
data_y_prueb = data_y[-50:]


##### Crear el model de Regresion Lineal #####
# Simple
modelS = LinearRegression()

# Cruzado
modelC = LinearRegression()


##### Entrenamos del modelo #####
# Simple
modelS.fit(data_x_entre, data_y_entre)


##### Predicciones #####
# Simple
predic = modelS.predict(data_x_prueb)

# Cruzada
predic_cross = cross_val_predict(modelC, data_xT, data_y, cv=10)


##### Score #####
# Simple
scor = modelS.score(data_x_prueb, data_y_prueb)

# Cruzada
scor_cross = cross_val_score(modelC,data_xT, data_y,cv=10).mean()


##### Error cuadratico #####
# Simple
cuadr = np.mean((predic - data_y_prueb) ** 2)

# Cruzado
cuadr_cross = np.mean((predic_cross - data_y) ** 2)


##### Resumen de los ajustes del modelo #####
# Simple
print '\n******** Entrenamiento Simple ********'
# El coeficiente
print 'Coeficiente: %.4f'%modelS.coef_
# Score: 1 es la prediccion perfecta
print 'Score: %.2f'%scor
# Los errores cuadraticos
print "Suma residual de cuadrados: %.2f"%cuadr

# Cruzado
print '\n******** Entrenamiento 10-Cruzado ********'
# El coeficiente
print 'Coeficiente: %.4f'%(1.*(predic_cross[1]-predic_cross[0])/(data_x_prueb[1]-data_x_prueb[0]))
# Score: 1 es la prediccion perfecta
print 'Score: %.2f'%scor_cross
# Los errores cuadraticos
print "Suma residual de cuadrados: %.2f"%cuadr_cross


##### Ploteando #####
# Simple
fig1,ax = plt.subplots()
ax.plot(data_x_prueb, predic, 'r^', lw=4)
ax.scatter(data_x_prueb,data_y_prueb,color='green')
ax.set_xlabel('Variable %i'%colx)
ax.set_ylabel('Variable %i'%coly)
ax.set_title('Comparacion Simple\nVariable %i vs. Variable %i'%(colx,coly))
fig1.savefig('Compa_Simple.jpg')

# Cruzado
fig2,ay = plt.subplots()
ay.plot(data_x_prueb, predic_cross[-50:], 'r^', lw=4)
ay.scatter(data_x_prueb,data_y_prueb)
ay.set_xlabel('Variable %i'%colx)
ay.set_ylabel('Variable %i'%coly)
ay.set_title('Comparacion Cruzada\nVariable %i vs. Variable %i'%(colx,coly))
fig2.savefig('Compa_Cruzada.jpg')

plt.show()
