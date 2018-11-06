import pandas as pd
import subprocess
import matplotlib.pylab as plt
import numpy as np

#Descargo el archivo de la web por si no esta

#subprocess.call("wget http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat", shell=True)

#Los nombres se dieron de acuerdo a la descripcion de los datos. 1 es el valor promedio, 2 la STD y 3 es outlier.

df = pd.read_csv("WDBC.dat", names = ['id','diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1', 'concavity1', 'cpoints1', 'symmetry1', 'fractald1','radius2','texture2','perimeter2','area2','smoothness2','compactness2', 'concavity2', 'cpoints2', 'symmetry2', 'fractald2','radius3','texture3','perimeter3','area3','smoothness3','compactness3', 'concavity3', 'cpoints3', 'symmetry3', 'fractald3'])

# Hago un analisis de todos los parametros (todas_medidas), pero tambien hago analisis de solo los 10 primeros.

todas_medidas = ['radius1','texture1','perimeter1','area1','smoothness1','compactness1', 'concavity1', 'cpoints1', 'symmetry1', 'fractald1','radius2','texture2','perimeter2','area2','smoothness2','compactness2', 'concavity2', 'cpoints2', 'symmetry2', 'fractald2','radius3','texture3','perimeter3','area3','smoothness3','compactness3', 'concavity3', 'cpoints3', 'symmetry3', 'fractald3']
medidas= ['radius1','texture1','perimeter1','area1','smoothness1','compactness1', 'concavity1', 'cpoints1', 'symmetry1', 'fractald1']

m = df.loc[:,medidas].values # Tomo solo los datos de mediciones
mT = df.loc[:,todas_medidas].values

def tratar(data): # Hago funcion que trate los datos para PCA de forma automatica
	f = data.shape[0]
	c = data.shape[1]
	resp = np.ones([f,c])

	for i in range(c):
		mean = np.mean(data[:,i])
		resp[:,i] = data[:,i] - mean
		std = np.std(resp[:,i])
		resp[:,i] = resp[:,i]/std
	return resp
#

def cov(data): # Calculo matriz de covarianza. Las columnas corresponden a distintos observables y las filas a las mediciones de estos.
	repeticiones = data.shape[0]
	c = data.shape[1]
	covr = np.ones([c, c])
	for i in range(c):
		for j in range(c):
			covr[i,j] = np.sum(data[:,i] * data[:,j]) / (repeticiones -1)
	return covr

# Tratamos primero los datos

malignosDF = df[df['diagnosis']=='M']
malignos = malignosDF.loc[:,medidas].values # Datos para celulas malignas
norm_malignos = tratar(malignos)

benignosDF = df[df['diagnosis']=='B'] 
benignos = benignosDF.loc[:,medidas].values # Datos para celulas benignas
norm_benignos = tratar(benignos)

malignosDFT = df[df['diagnosis']=='M']
malignosT = malignosDFT.loc[:,todas_medidas].values
norm_malignosT = tratar(malignosT)

benignosDFT = df[df['diagnosis']=='B']
benignosT = benignosDFT.loc[:,todas_medidas].values
norm_benignosT = tratar(benignosT)

m1 = tratar(m)
m1T = tratar(mT)
# calculamos la matriz de covarianza

cov_m1 = cov(m1)
cov_m1T = cov(m1T)


eigenvals = np.linalg.eig(cov_m1)[0] # Tomo los autovectores con mayores autovalores.
eigenvecs = np.linalg.eig(cov_m1)[1]

eigenvalsT = np.linalg.eig(cov_m1T)[0]
eigenvecsT = np.linalg.eig(cov_m1T)[1]


for i in range(eigenvals.size):
	print "----------------------------------","\nValor propio ",i+1,": ", eigenvals[i], "\n","Vector propio ",i+1,": \n", eigenvecs[i], "\n----------------------------------"

pc1 = eigenvecs[0]
pc2 = eigenvecs[1]

b_proy_1 = np.dot(norm_benignos,pc1)
b_proy_2 = np.dot(norm_benignos,pc2)

m_proy_1 = np.dot(norm_malignos,pc1)
m_proy_2 = np.dot(norm_malignos,pc2)

plt.figure()
plt.scatter(b_proy_1,b_proy_2, color="blue", label="benignos")
plt.scatter(m_proy_1,m_proy_2, color="red", label = "malignos")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.savefig("CorrealSergio_PCA.pdf")
plt.close()

# Ahora grafico teniendo en cuenta todos los parametros

pc1T = eigenvecsT[0]
pc2T = eigenvecsT[1]

b_proy_1T = np.dot(norm_benignosT,pc1T)
b_proy_2T = np.dot(norm_benignosT,pc2T)

m_proy_1T = np.dot(norm_malignosT,pc1T)
m_proy_2T = np.dot(norm_malignosT,pc2T)

plt.figure()
plt.scatter(b_proy_1T,b_proy_2T, color="blue", label="benignos")
plt.scatter(m_proy_1T,m_proy_2T, color="red", label = "malignos")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.savefig("CorrealSergio_PCA_NoConcluyente.pdf")
plt.close()

print "Los parametros mas importantes son maximo 5. Las componentes de mayor magnitud del primer autovector indican que el radio, la textura y la compactez son los parametros mas importantes. Las componentes del segundo autovector indican que, en menor medida, el perimetro y la suavidad tambien son determinantes.\n\nLa figura CorrealSergio_PCA muestra la proyeccion de los datos solo teniendo en cuenta los promedios de los parametros (i.e. 10 variables). Esta figura muestra que con PCA no es posible diferenciar celulas malignas y benignas porque se encuantran en regiones equivalentes de la grafica. Incluso, si se toman las 30 variables iniciales resulta la figura CorrealSergio_PCA_NoConcluyente que muestra que tener en cuanta todos estos datos no permite tambpoco concluir sobre el estado de las celulas.\n"