import numpy as np
import matplotlib.pyplot as plt

t,target,f0,f1,f2,f3,f4,f5,f6,f7,f8=np.loadtxt('dataset_11.dat', skiprows=1, unpack=True)
N=np.size(f0)
F=np.zeros((N,9))
cov=np.zeros((9,9))

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

featvec=eigvec[:,0:9]

finaldata=np.transpose(np.dot(np.transpose(featvec),np.transpose(F)))

print eigval	

plt.plot(t,finaldata[:,0])
plt.plot(t,finaldata[:,1])
plt.plot(t,finaldata[:,2])
plt.plot(t,finaldata[:,3])
plt.plot(t,finaldata[:,4])
plt.plot(t,finaldata[:,5])
plt.plot(t,finaldata[:,6])
plt.plot(t,finaldata[:,7])
plt.plot(t,finaldata[:,8])

plt.plot(t,F[:,0],'+')

plt.plot()
plt.show()