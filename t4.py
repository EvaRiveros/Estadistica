import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold



############################  P1  ################################
#Retorna vector de parametros y grafica ajuste a los datos con p=5
t,y=np.loadtxt('datos.dat',skiprows=1,usecols=(0,1),unpack=True) #t_k, Y_k

p=5 #grado polinomio
Ti=0.4
Tf=0.7
sigma=30*10**(-6)

# MX=V

def transito(x):
	if (Ti<x and x<Tf):
		return 1
	return 0

def matriz(t,y,p,n):
	M=np.zeros((p+2,p+2))

	for k in xrange(0,n):
		M[0][0]+=transito(t[k])

	for m in xrange(1,p+2):
		for k in xrange(0,n):
			M[m][0]+= (t[k]**(m-1)) * transito(t[k])
		M[0][m]=-M[m][0] 

	for m in xrange(1,p+2):
		for l in xrange(1,p+2):
			for k in xrange(0,n):
				M[m][l]+=( -(t[k]**(m-1)) * (t[k]**(l-1)) )
	return M

def vector(t,y,p,n):
	V=np.zeros(p+2)	
	for k in xrange(0,n):
		V[0]+= (1-y[k])*transito(t[k])

	for m in xrange(1,p+2):
		for k in xrange(0,n):
			V[m]+= (1-y[k])*(t[k]**(m-1))
	return V

M=matriz(t,y,p,np.size(y))
V=vector(t,y,p,np.size(y))


X=np.linalg.inv(M).dot(V.T) #X=M^(-1)V

def Modelo(time,power):
	suma=1-X[0]*transito(time)
	for i in xrange(0,power+1):
		suma+=X[i+1]*(time**(i))
	return suma


th=np.linspace(np.amin(t),np.amax(t),400)

H=np.zeros(400)
for i in xrange(0,400):
	H[i]=Modelo(th[i],p)


plt.plot(t,y,'.')
plt.plot(th,H)
plt.xlabel('Tiempo')
plt.ylabel('Flujo')
plt.title('Ajuste para p=5')
plt.show()

print 'Vector de parametros (delta, c0,...,cp):',X


############################  P3a  ################################
#Ocupar AIC y BIC para encontrar el grado optimo del polinomio
aic=np.zeros(9)
bic=np.zeros(9)
P=np.array([1,2,3,4,5,6,7,8,9])

for i in xrange(1,10): #i son los distintos grados del pol.
	M=matriz(t,y,i,np.size(y))
	V=vector(t,y,i,np.size(y))
	X=np.linalg.inv(M).dot(V.T)

	for j in xrange(0,np.size(y)):
		H[j]=Modelo(t[j],i)

	suma=0
	for k in xrange(0,np.size(y)):
		suma+=((y[k]-H[k])**2)/(2.0*sigma**2)

	ln=(np.size(y)/2)*math.log(1/(2*np.pi*sigma**2))-suma

	aic[i-1]=2*(-ln + (i+2) + ((i+2.0)*(i+3.0))/(np.size(y)-i+1.0))
	bic[i-1]=-2*ln+ (i+2)*math.log(np.size(y))

plt.plot(P,aic,'b')
plt.plot(P,bic,'r')
plt.text(8, -5313, 'BIC')
plt.text(8, -5351, 'AIC')
plt.xlabel('Grado polinomio')
plt.show()


############################  P3b  ################################
#Usar K-fold CV para encontrar el grado optimo del polinomio
K=4 #K-fold
kf=KFold(np.size(y),K,shuffle=True)
cve=np.zeros(np.size(P)) #cada elemento de cve es el promedio de los K cvek
						 #para cada grado del polinomio
for grado in P:
	cvek=0	
	for train, test in kf:
		Mk=matriz(t[train],y[train],grado,np.size(t[train]))
		Vk=vector(t[train],y[train],grado,np.size(t[train]))
		X=np.linalg.inv(Mk).dot(Vk.T)
		H=np.zeros(np.size(y[test]))

		for i in xrange(0,np.size(t[test])):
			H[i]=Modelo(t[test][i],grado)

		for k in xrange(0,np.size(y[test])):
			cvek+=(y[test][k]-H[k])**2

	cve[grado-1]=cvek/K

plt.plot(P,cve,'b')
plt.xlabel('Grado polinomio')
plt.title('K-fold cross validation con K='+str(K))
plt.show()


