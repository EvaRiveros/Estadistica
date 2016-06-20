import numpy as np
import matplotlib.pyplot as plt
import batman
from pymc3 import Model, Normal

t,target,f0,f1,f2,f3,f4,f5,f6,f7,f8=np.loadtxt('dataset_11.dat', skiprows=1, unpack=True)
N=np.size(f0)
p=9 #numero de dimensiones con las que nos quedamos despues de PCA

##################################### PCA ###############################################

F=np.zeros((N,9))
cov=np.zeros((9,9))

target=np.log(target)-np.mean(np.log(target))
F[:,0]=np.log(f0)-np.mean(np.log(f0))
F[:,1]=np.log(f1)-np.mean(np.log(f1))
F[:,2]=np.log(f2)-np.mean(np.log(f2))
F[:,3]=np.log(f3)-np.mean(np.log(f3))
F[:,4]=np.log(f4)-np.mean(np.log(f4))
F[:,5]=np.log(f5)-np.mean(np.log(f5))
F[:,6]=np.log(f6)-np.mean(np.log(f6))
F[:,7]=np.log(f7)-np.mean(np.log(f7))
F[:,8]=np.log(f8)-np.mean(np.log(f8))

for i in xrange(0,9):
    for j in xrange(0,9):
        cov[i,j]=np.cov(F[:,i],F[:,j])[0,1]
eigval,eigvec=np.linalg.eig(cov)

eigval=np.sort(eigval)[::-1] #ordenar valores propios de mayor a menor
eigvec[:,[4,5]]=eigvec[:,[5,4]] #cambiar el orden de los vectores propios para que coincidan con el de los valores
eigvec[:,[6,8]]=eigvec[:,[8,6]]

featvec=eigvec[:,0:p]

finaldata=np.dot(np.transpose(featvec),np.transpose(F))

finaldatanorm=np.zeros(np.shape(finaldata))
for i in xrange(0,p):
	finaldatanorm[i,:]=finaldata[i,:]/np.linalg.norm(finaldata[i,:])



################################ coeficientes ###########################################

A=np.dot(finaldata,F) # A es de px9, coeficientes
for i in xrange(0,p):
	A[i,:]=A[i,:]/(np.linalg.norm(finaldata[i,:]))**2



# plt.plot(t,np.dot(A[:,0],finaldata)) #modelo log(flujo) estrella 0
# plt.plot(t,F[:,0],'+') #log(flujo) real estrella 0 

# print valores propios

print eigval

# plotear componentes S

plt.plot(t,finaldata[0,:])
plt.plot(t,finaldata[1,:])
plt.plot(t,finaldata[2,:])
plt.plot(t,finaldata[3,:])
plt.plot(t,finaldata[4,:])
plt.plot(t,finaldata[5,:])
plt.plot(t,finaldata[6,:])
plt.plot(t,finaldata[7,:])
plt.plot(t,finaldata[8,:])

plt.plot()
plt.show()


########################### estimacion constante ########################################

atarget=np.mean(A,axis=1) #estimacion coeficientes alfa para la estrella objetivo
t_st=t[np.logical_or(t>1.,t<-1.)] #vector tiempo sin transito
target_st=target[np.logical_or(t>1.,t<-1.)] #flujo estrella objetivo sin transito

c_est=np.mean(target_st-np.dot(atarget,finaldata)[np.logical_or(t>1.,t<-1.)]) #0.007+-0.05

#########################################################################################


params = batman.TransitParams()
params.t0 = 0.                       #time of inferior conjunction
params.per = 0.78884                 #orbital period
params.rp = rp                       #planet radius (in units of stellar radii)
params.a = a                         #semi-major axis (in units of stellar radii)
params.inc = i                       #orbital inclination (in degrees)
params.ecc = 0.                      #eccentricity
params.w = 90.                       #longitude of periastron (in degrees)
params.u = [0.1, 0.3]                #limb darkening coefficients
params.limb_dark = "quadratic"       #limb darkening model

t = np.linspace(-2,2.5,100)

m = batman.TransitModel(params,t)
flux = m.light_curve(params)


#########################################################################################

basic_model=Model()

with basic_model:
	# priors parametros
	c=Normal('constante',mu=c_est,sd=0.05)
	
	

