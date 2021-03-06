import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

###################### Regresion lineal ##############################
#P=Mb
#M matriz con posiciones (x,y) y unos (para incluir el intercepto en b)
#b vector 3D de parametros del plano

x,y,tag=np.loadtxt("datos_clasificacion.dat", skiprows=1, unpack=True)

tag=tag-1
u=0.5 #threshold
X=np.zeros((np.size(x),2))
X[:,0]=x
X[:,1]=y

x0=x[tag==0]
x1=x[tag==1]
y0=y[tag==0]
y1=y[tag==1]

M=np.zeros((np.size(x),3))
b=np.zeros(3)

M[:,0]=1
M[:,1]=x
M[:,2]=y

Mt=np.transpose(M)
inv=np.linalg.inv(np.dot(Mt,M))

b=np.dot(np.dot(inv,Mt),tag)

P=np.dot(M,b)

xrecta=np.linspace(-4,12,50)
yrecta=(u-b[0])/b[2]-(b[1]/b[2])*xrecta

pred_RL=np.ones(np.size(P))
pred_RL[P<u]=0

############################### LDA ##################################

#Estimamos prior p, media mu, matriz de cov sigma 

p=np.zeros(2)
mu=np.zeros((2,2))
sigma=np.zeros((2,2))
N=np.size(x) #numero total de datos
N0=np.size(x0) #numero de elementos clase 0
N1=np.size(x1) #numero de elementos clase 1

p[0]=(N0*1.0)/N
p[1]=(N1*1.0)/N

suma=0
for i in xrange(0,N0):
	suma+=x0[i]
mu[0,0]=suma/N0 #mu_0x

suma=0
for i in xrange(0,N0):
	suma+=y0[i]
mu[0,1]=suma/N0 #mu_0y

suma=0
for i in xrange(0,N1):
	suma+=x1[i]
mu[1,0]=suma/N1 #mu_1x

suma=0
for i in xrange(0,N1):
	suma+=y1[i]
mu[1,1]=suma/N1 #mu_1y


suma0=0
suma1=0
for i in xrange(0,N0):
	suma0+=(x0[i]-mu[0,0])**2
for i in xrange(0,N1):
	suma1+=(x1[i]-mu[1,0])**2
sigma[0,0]=(suma0+suma1)/(N-2)


suma0=0
suma1=0
for i in xrange(0,N0):
	suma0+=(y0[i]-mu[0,1])**2
for i in xrange(0,N1):
	suma1+=(y1[i]-mu[1,1])**2
sigma[1,1]=(suma0+suma1)/(N-2)


suma0=0
suma1=0
for i in xrange(0,N0):
	suma0+=(x0[i]-mu[0,0])*(y0[i]-mu[0,1])
for i in xrange(0,N1):
	suma1+=(x1[i]-mu[1,0])*(y1[i]-mu[1,1])
sigma[1,0]=sigma[0,1]=(suma0+suma1)/(N-2)

a=mu[0,:]+mu[1,:]
b=mu[0,:]-mu[1,:]
c=np.linalg.inv(sigma)

inter= (-np.log(p[0]/p[1])+0.5*np.dot(a,np.dot(c,b)))/np.dot(np.linalg.inv(sigma),mu[0,:]-mu[1,:])[1]

pend=-np.dot(np.linalg.inv(sigma),mu[0,:]-mu[1,:])[0]/np.dot(np.linalg.inv(sigma),mu[0,:]-mu[1,:])[1]

xlda=np.linspace(-4,12,50)
ylda=pend*xlda+inter


############################### QDA ##################################

sigma1=np.zeros((2,2))
sigma0=np.zeros((2,2))

def d0(array):
	return -0.5*(np.log(np.linalg.det(sigma0))+np.dot(np.transpose(array-mu[0,:]),np.dot(np.linalg.inv(sigma0),array-mu[0,:])))+np.log(p[0])

def d1(array):
	return -0.5*(np.log(np.linalg.det(sigma1))+np.dot(np.transpose(array-mu[1,:]),np.dot(np.linalg.inv(sigma1),array-mu[1,:])))+np.log(p[1])

suma0=0
suma1=0
for i in xrange(0,N0):
	suma0+=(x0[i]-mu[0,0])**2
for i in xrange(0,N1):
	suma1+=(x1[i]-mu[1,0])**2
sigma0[0,0]=suma0/N0
sigma1[0,0]=suma1/N1


suma0=0
suma1=0
for i in xrange(0,N0):
	suma0+=(y0[i]-mu[0,1])**2
for i in xrange(0,N1):
	suma1+=(y1[i]-mu[1,1])**2
sigma0[1,1]=suma0/N0
sigma1[1,1]=suma1/N1


suma0=0
suma1=0
for i in xrange(0,N0):
	suma0+=(x0[i]-mu[0,0])*(y0[i]-mu[0,1])
for i in xrange(0,N1):
	suma1+=(x1[i]-mu[1,0])*(y1[i]-mu[1,1])
sigma0[1,0]=sigma0[0,1]=suma0/N0
sigma1[1,0]=sigma1[0,1]=suma1/N1


xlin=np.linspace(0,10,100)
ylin=np.linspace(0,10,100)

xqda=np.array([0])
yqda=np.array([0])
k=0

for i in xrange(0,np.size(xlin)):
	for j in xrange(0,np.size(xlin)):
		a=np.array([xlin[i],ylin[j]])
		if (math.fabs(d0(a)-d1(a))<0.1): #decision boundary entre las clases 0 y 1
			if (k==0):
				xqda[0]=a[0]
				yqda[0]=a[1]
				k+=1
			xqda=np.append(xqda,a[0])
			yqda=np.append(yqda,a[1])

pred_QDA=np.ones(np.size(P))
for x in xrange(0,N):
	if (d0(X[x])>d1(X[x])):
		pred_QDA[x]=0

plt.plot(x0,y0,'o')
plt.plot(x1,y1,'o')
plt.plot(xrecta,yrecta)
plt.plot(xqda,yqda,'+')
plt.plot(xlda,ylda,'k^')
plt.title('Decision boundaries para RL (recta), LDA (triangulos), y QDA (curva cian)')
plt.plot()
plt.show()

######################################################################

neigh = KNeighborsClassifier(n_neighbors=3) 
neigh.fit(X,tag) 
KNeighborsClassifier()
pred_KN=neigh.predict(X)

######################################################################

clf = svm.SVC()
clf.fit(X,tag)
pred_SVC=clf.predict(X)

######################################################################


def confusion(modelo,real):
	m=np.zeros((2,2))
	m[0,0]=sum(np.logical_and(modelo==0,real==0))
	m[1,1]=sum(np.logical_and(modelo==1,real==1))
	m[1,0]=sum(np.logical_and(modelo==0,real==1))
	m[0,1]=sum(np.logical_and(modelo==1,real==0))
	return m

def MR(matriz):
	return (matriz[1,0]+matriz[0,1])/N

c1=confusion(pred_RL,tag)
c2=confusion(pred_QDA,tag)
c3=confusion(pred_KN,tag)
c4=confusion(pred_SVC,tag)

print "Matrices de confusion y Misclassification Rate (MR):"
print "-Regresion lineal y LDA\n",c1
print "MR=", MR(c1)
print "-QDA\n",c2
print "MR=", MR(c2)
print "-Nearest neighbors (3)\n",c3
print "MR=", MR(c3)
print "-Support vector machines (RBF)\n",c4
print "MR=", MR(c4)


######################################################################

sigmanuevo1=np.zeros((2,2))
sigmanuevo2=np.zeros((2,2))
munuevo1=np.array([2,3])
munuevo2=np.array([6,6])

sigmanuevo1[0,0]=5
sigmanuevo1[1,0]=-2
sigmanuevo1[0,1]=-2
sigmanuevo1[1,1]=5

sigmanuevo2[0,0]=1
sigmanuevo2[1,0]=0
sigmanuevo2[0,1]=0
sigmanuevo2[1,1]=1

xreal=np.array([0])
yreal=np.array([0])
k=0

for i in xrange(0,np.size(xlin)):
	for j in xrange(0,np.size(xlin)):
		a=np.array([xlin[i],ylin[j]])
		delta1=-0.5*(np.log(np.linalg.det(sigmanuevo1))+np.dot(np.transpose(a-munuevo1),np.dot(np.linalg.inv(sigmanuevo1),a-munuevo1)))+np.log(p[0])
		delta2=-0.5*(np.log(np.linalg.det(sigmanuevo2))+np.dot(np.transpose(a-munuevo2),np.dot(np.linalg.inv(sigmanuevo2),a-munuevo2)))+np.log(p[1])

		if (math.fabs(delta1-delta2)<0.1):
			if (k==0):
				xreal[0]=a[0]
				yreal[0]=a[1]
				k+=1
			xreal=np.append(xreal,a[0])
			yreal=np.append(yreal,a[1])

pred_QDAR=np.ones(np.size(P))
for x in xrange(0,N):
	delta1=-0.5*(np.log(np.linalg.det(sigmanuevo1))+np.dot(np.transpose(X[x]-munuevo1),np.dot(np.linalg.inv(sigmanuevo1),X[x]-munuevo1)))+np.log(p[0])
	delta2=-0.5*(np.log(np.linalg.det(sigmanuevo2))+np.dot(np.transpose(X[x]-munuevo2),np.dot(np.linalg.inv(sigmanuevo2),X[x]-munuevo2)))+np.log(p[1])

	if (delta1>delta2):
		pred_QDAR[x]=0


plt.plot(x0,y0,'o')
plt.plot(x1,y1,'o')
plt.plot(xrecta,yrecta)
plt.plot(xqda,yqda,'+')
plt.plot(xlda,ylda,'k^')
plt.plot(xreal,yreal,'m*')
plt.title('Decision boundaries para RL (recta), LDA (triangulos), QDA (curva cian), y modelo real (parte c, * magentas)')
plt.plot()
plt.show()

c5=confusion(pred_QDAR,tag)
print "-QDA con modelo real\n",c5
print "MR=", MR(c5)