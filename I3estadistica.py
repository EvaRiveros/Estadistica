import numpy as np
import matplotlib.pyplot as plt

t,target,s0,s1,s2,s3,s4,s5,s6,s7,s8=np.loadtxt('dataset_11.dat', skiprows=1, unpack=True)
N=np.size(s0)
S=np.zeros((N,N))
cov=np.zeros((9,9))

S[:,0]=s0-np.mean(s0)
S[:,1]=s1-np.mean(s1)
S[:,2]=s2-np.mean(s2)
S[:,3]=s3-np.mean(s3)
S[:,4]=s4-np.mean(s4)
S[:,5]=s5-np.mean(s5)
S[:,6]=s6-np.mean(s6)
S[:,7]=s7-np.mean(s7)
S[:,8]=s8-np.mean(s8)

for i in xrange(0,9):
    for j in xrange(0,9):
        cov[i,j]=np.cov(S[:,i],S[:,j])[0,1]
eigval,eigvec=np.linalg.eig(cov)

print eigvec

eigval=np.sort(eigval)[::-1]
eigvec[:,[4,5]]=eigvec[:,[5,4]]
eigvec[:,[6,8]]=eigvec[:,[8,6]]

print eigvec



m=np.zeros((9,np.size(s1))) #matriz para los coeficientes a_ij


# plt.plot(t,s0)
# plt.plot(t,S[:,0])
# # plt.plot(t,s3)
# # plt.plot(t,s4)
# # plt.plot(t,s5)
# plt.plot()
# plt.show()
