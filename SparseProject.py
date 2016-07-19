import numpy as np
import scipy as sio
import scipy.io
import matplotlib.pyplot as plt
import pylab
from StimSet import StimSet

IMG=scipy.io.loadmat('IMAGES')
Pic_array=IMG['IMAGES']

inftime=70
buffr=20
batch_size=100
basis_size=196 # I chose 200 phi
data_dim=256 #16x16 flattened
X=np.zeros([batch_size,data_dim])

eta=.01
lambd=.15 #lambda/sigma
ntrials=200
alpha=.1

"""randomly initializes and normalizes the phi matrix"""
phi = np.random.randn(basis_size, data_dim)
sq=phi*phi
total=np.sqrt(np.sum(sq,axis=0))
phi=phi/total


costX1=np.zeros(inftime)
costY1=np.zeros(inftime)
costX2=np.zeros(ntrials)
costY2=np.zeros(ntrials)

# main loop
for i in range(ntrials):
    
    acts=np.zeros([basis_size,batch_size])
    
    #Randomly cuts out a 16x16x1 matrix from the given data.
    #I make sure that the edges(20pixels) are not included.
    
    
    for j in range(batch_size):

        which = np.random.choice(10)
        row=buffr+np.random.choice(472)
        col=buffr+np.random.choice(472)
        temp=Pic_array[row:(row+16),col:(col+16),which]
        temp=temp-np.mean(temp)
        temp=temp/np.std(temp)
        temp=temp.flatten()
        X[j]=temp
        
        
    for k in range(inftime):
    
        phi_sq=np.dot(phi,phi.T)
        da_dt=np.dot(phi,X.T)-np.dot(phi_sq,acts)-lambd*acts*(1/(1+acts*acts))
        acts=(1-eta)*acts+eta*(da_dt)
        
        #tabulate the cost for inference
        costX1[k]=k
        costY1[k]=np.mean((X-np.dot(acts.T,phi))**2)
    
    
    
    
           
    #update phi
    d_phi=np.dot(acts,(X-np.dot(acts.T,phi))) 
    d_phi=np.mean(d_phi,axis=0)
    phi=phi+alpha*d_phi
    tot=np.sqrt(np.sum(phi*phi,axis=0))
    phi= phi/tot
    
    #tabulate the cost after the gradient decent
    costX2[i]=i
    costY2[i]=np.mean((X-np.dot(acts.T,phi))**2)
    
    print costY2[i]
    print i

    
please_work=StimSet(X,(16,16))
Array=please_work._stimarray(phi,(16,16))

print costX2
print costY2
plt.figure(0) 
plt.imshow(Array,'gray')
pylab.show()

plt.figure(1)        
plt.plot(costX2, costY2, 'ro')
plt.show()

plt.figure(3)        
plt.plot(costX1, costY1, 'ro')
plt.show()