import numpy as np
from random import randrange

#data=np.loadtxt('toyexample.data',dtype=int)
print('start')
data=np.loadtxt('nips.data',dtype=int)
print('read data done')
data[:,0]-=1
data[:,1]-=1

vocab=open('nips.vocab').read().split('\n')[0:-1]
print('vocab is done')
dic=dict(zip(range(0,len(vocab)+1),vocab))
[di,wi,ci,citest]=data.transpose()
D=np.max(di)+1
W=np.amax(wi)+1
citest = citest.transpose()

K=10
print('start to set DW')
DW=[[] for i in range(0,D)]
for i in data:
    DW[i[0]].extend([i[1]]*i[2])


Nd=[len(i) for i in DW]

Z_init=[[randrange(0,K) for i in range(0,Nd[doc])] for doc in range(0,D)]

Adk=np.zeros((D,K))
for dnum,d in enumerate(Z_init):
    for z in d:
        Adk[dnum][z]+=1


Bkw=np.zeros((K,W))
for dnum,d in enumerate(Z_init):
    for wnum,z in enumerate(d):
        Bkw[z][DW[dnum][wnum]]+=1


Mk=Bkw.sum(axis=1)

phi=np.zeros((K,W))
alpha=0.1
beta=0.1
iteration=100
Kalpha=alpha*K
Wbeta=beta*W

print ('Read data finish!')

def Collapsed_gibbs_sampling(Z,d,w_local_id):
    w=DW[d][w_local_id]
    z=Z[d][w_local_id]
    
    Adk[d][z]-=1
    Bkw[z][w]-=1
    Mk[z]-=1
    Nd[d]-=1
    P=np.zeros((K))
    
    for k in range(0,K):
        P[k]=((beta+Bkw[k][w])/(Mk[k]+Wbeta))*((alpha+Adk[d][k])/(Nd[d]+Kalpha))
    P=P/np.sum(P)
    new_z=np.random.choice(K,p=P)
    Z[d][w_local_id]=new_z
    
    Adk[d][new_z]+=1
    Bkw[new_z][w]+=1
    Mk[new_z]+=1
    Nd[d]+=1
    
    return Z

def compute_theta():
    theta=np.zeros((D,K))
    for d in range(0,D):
        for k in range(0,K):
            theta[d][k]=(Adk[d][k]+alpha)/(Nd[d]+Kalpha)
    return theta

def compute_phi():
    phi=np.zeros((K,W))
    for k in range(0,K):
        for w in range(0,W):
            phi[k][w]=(Bkw[k][w])/(Mk[k]+Wbeta)
    return phi



def LDA():
    Z=Z_init
    for i in range(0,iteration+1):
        for d in range(0,D):
            for w_local_id in range(0,Nd[d]):
                print('iteration,document,word:',i,d,w_local_id)
                Z=Collapsed_gibbs_sampling(Z,d,w_local_id)
    theta=compute_theta()
    phi=compute_phi()
    return theta,phi
    
    
theta,phi=LDA()
for k in range(0,K):
    print ('Topic:',k)
    sort_w=list(reversed(np.argsort(phi[k])))
    for w in sort_w[0:20]:
        if phi[k][w]>0:
            print(phi[k][w],dic[w])
