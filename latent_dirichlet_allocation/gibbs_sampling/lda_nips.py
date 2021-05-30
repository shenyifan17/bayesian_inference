import numpy as np
from random import randrange


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


def collapsed_loglikeli():
    likeli=np.sum(np.log(gamma(Bkw+beta)))+np.sum(np.log(gamma(Adk+alpha)))\
    -np.sum(np.log(gamma(np.asarray(Mk)+Wbeta)))-np.sum(np.log(gamma(np.asarray(Nd)+Kalpha)))
    return likeli


def gipps_loglikeli():
    likeli=np.sum((Bkw+beta-1)*np.log((Bkw+beta)/np.transpose(np.tile(np.asarray(Mk+Wbeta),(W,1)))))\
    +np.sum((Adk+alpha-1)*np.log((Adk+alpha)/np.transpose(np.tile((np.asarray(Nd)+Kalpha),(K,1)))))
    return likeli

def CollapseGibbs_LDA():
    likeli=[]
    pred=[]
    Z=Z_init
    for i in range(0,iteration):
        print ('iteration:',i)
        for d in range(0,D):
            for w_local_id in range(0,Nd[d]):
                Z=Collapsed_gibbs_sampling(Z,d,w_local_id)
        likeli.append(collapsed_loglikeli())
        theta=compute_theta()
        phi=compute_phi()
        pred.append(predict(theta,phi))

    theta=compute_theta()
    phi=compute_phi()
    return theta,phi,likeli,pred

def predict(theta,phi):
    Pi=np.zeros((I,1))
    for i in range(0,I):
        Pi[i]=np.dot(theta[di[i],:],phi[:,wi[i]])
    #citest_t=citest.reshape(1,I)
    log_Pi=np.log(Pi)
    P=np.dot(citest,log_Pi)
    return P[0]

def Gibbs_LDA():
    likelihood=[]
    pred=[]
    Z=Z_init
    theta=compute_theta()
    phi=compute_phi()
    for i in range(0,iteration):
        for d in range(0,D):
            for w_local_id in range(0,Nd[d]):
                P=np.zeros((K))
                w=DW[d][w_local_id]
                z=Z[d][w_local_id]
                for k in range(0,K):
                    P[k]=phi[k][w]*theta[d][k]
                P=P/np.sum(P)
                new_z=np.random.choice(K,p=P)
                Adk[d][z]-=1
                Bkw[z][w]-=1
                Mk[z]-=1
                Adk[d][new_z]+=1
                Bkw[new_z][w]+=1
                Mk[new_z]+=1
                Z[d][w_local_id]=new_z
#                 theta[d][new_z]=(alpha+Adk[d][k])/(Nd[d]+Kalpha)
#                 phi[new_z][w]=(beta+Bkw[k][w])/(Mk[k]+Wbeta)
        theta=compute_theta()
        phi=compute_phi()
        pred.append(predict(theta,phi))
        likelihood.append(gipps_loglikeli())
    return theta,phi,likelihood,pred




data=np.loadtxt('nips.data',dtype=int)
data[:,0]-=1
data[:,1]-=1
vocab=open('nips.vocab').read().split('\n')[0:-1]
dic=dict(zip(range(0,len(vocab)+1),vocab))
[di,wi,ci,citest]=data.transpose()
D=np.max(di)+1
W=np.amax(wi)+1

DW=[[] for i in range(0,D)]
for i in data:
    DW[i[0]].extend([i[1]]*i[2])

word_frequency=np.zeros((W))
doc_frequency=np.zeros((W,D))
for i in data:
    word_frequency[i[1]]+=i[2]
    if doc_frequency[i[1],i[0]]==0:
        doc_frequency[i[1],i[0]]=1


doc_frequency=np.sum(doc_frequency,axis=1)
tf_idf=word_frequency/doc_frequency
wordlist=range(0,W)
tosort_dict=dict(zip(wordlist, word_frequency.tolist()))
#tosort_dict=dict(zip(wordlist, tf_idf.tolist()))
sorted_dict=sorted(tosort_dict.items(), key=lambda x: x[1], reverse=True)
sorted_word=[i[0] for i in sorted_dict]

new_data=[]
for i in data:
    if tosort_dict[i[1]]>=147 or i[1]==4557:
        new_data.append(i)


new_data=np.array([l for l in new_data])

K=10
alpha=0.1
beta=0.1
iteration=100
Kalpha=alpha*K
Wbeta=beta*W

[di,wi,ci,citest]=new_data.transpose()
D=np.max(di)+1
W=np.amax(wi)+1
I=len(di)

DW=[[] for i in range(0,D)]
for i in new_data:
    DW[i[0]].extend([i[1]]*i[2])


Nd=[len(i) for i in DW]

Z_init=[[randrange(0,K) for i in range(0,Nd[doc])] for doc in range(0,D)]
saveZ_init=Z_init


for dnum,d in enumerate(Z_init):
    for z in d:
        Adk[dnum][z]+=1


Bkw=np.zeros((K,W))
for dnum,d in enumerate(Z_init):
    for wnum,z in enumerate(d):
        Bkw[z][DW[dnum][wnum]]+=1


Mk=Bkw.sum(axis=1)


theta,phi,likeli,pred=CollapseGibbs_LDA()

for k in range(0,K):
    print ('Topic:',k)
    sort_w=list(reversed(np.argsort(phi[k])))
    for w in sort_w[0:20]:
        if phi[k][w]>0:
            print(phi[k][w],dic[w])
