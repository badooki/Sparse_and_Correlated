#env XLA_PYTHON_CLIENT_MEM_FRACTION=.1
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsc
import copy
from sklearn.metrics import pairwise_distances
import mnist
import matplotlib.pyplot as plt
from jax import random
import sys
from os.path import exists
import os
from jax import jit


def ReLU(A): return jnp.where(A>0,A,0)

def makeBand(N,bw): # bw: the number of additional 'diagonal' (one side). In other words, the actual bandwidth would be 2m+1
    return jnp.abs(jnp.arange(N)[:,None]-jnp.arange(N)[None,:])<(bw+1)

def getCircleX(Ns,N):
    anglesr=jnp.linspace(0,2*jnp.pi,Ns)
    circle2d=jnp.block([[jnp.cos(anglesr)[None,:]], [jnp.sin(anglesr)[None,:]], [jnp.zeros((N-2,Ns))]])
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    M=random.normal(key,(N,N))
    U,_,_=jnp.linalg.svd(M)
    X=jnp.matmul(U,circle2d).T
    return X

def loadTauBound(home):
    s=np.load(home+'mypylib/tauBound.npz')
    sigmas=s['sigmas']
    tauBound=s['tauBound']
    def getTauBound(D):
        _,yd=lookupy(D,x=sigmas,y=tauBound)
        return yd
    return getTauBound

def loadFixedPoints(home):
    s=np.load(home+'mypylib/fixedPointsRaw.npz')
    sigmas=s['sigmas']
    taus=s['taus']
    Fcs=s['Fcs']
    def getConvBound(D,thres=0.1): #threshold for t #jnp.nanmin(yd)==yd
        mask=Fcs<thres
        tauBound_conv=taus[np.argmax(np.abs(np.gradient(mask*1.0,axis=0)),axis=0)]
        _,yd=lookupy(D,x=sigmas,y=tauBound_conv)
        #if jnp.sum(jnp.nanmin(yd)==yd)>1:
        #    yd=jnp.where(jnp.nanmin(yd)==yd,jnp.nan,yd)
        return yd
    tauPowerBound=taus[jnp.nanargmax(Fcs,axis=0)]
    def getPowerBound(D):
        _,yd=lookupy(D,x=sigmas,y=tauPowerBound)
        return yd
    
    return getConvBound,getPowerBound,sigmas,taus,Fcs

def getSubclass(xtrainr_in, ytrainr_in, xtestr_in, ytestr_in , labels=[0,1],All=True):
    if All:
        return xtrainr_in, ytrainr_in, xtestr_in, ytestr_in
    else:
        train_mask=np.zeros(np.shape(ytrainr_in))
        for label in labels:
            train_mask=train_mask+(ytrainr_in==label)*1
        train_mask=train_mask==1
        
        test_mask=np.zeros(np.shape(ytestr_in))
        for label in labels:
            test_mask=test_mask+(ytestr_in==label)*1
        test_mask=test_mask==1
        
        return xtrainr_in[train_mask,:], ytrainr_in[train_mask], xtestr_in[test_mask,:], ytestr_in[test_mask]
    

def getFashion_train_or_test(home='', kind='train'):
    import os
    import gzip
    import numpy as np

    path=home+'mypylib/Data/fashion'
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

def getFashion(N_train=1000,N_test=1000,normalize=True,seed=0,home='',labels=[0,1],All=True):
    xtrainr, ytrainr = getFashion_train_or_test(home=home, kind='train')
    xtestr, ytestr = getFashion_train_or_test(home=home, kind='test')
    xtrainr, ytrainr, xtestr, ytestr  = getSubclass(xtrainr, ytrainr, xtestr, ytestr , labels=labels,All=All)
    return chooseRandomSubset(xtrainr,ytrainr,xtestr,ytestr,N_train=N_train,N_test=N_test,normalize=normalize,seed=seed)

def getMNIST(N_train=1000,N_test=1000,normalize=True,seed=0,labels=[0,1],All=True):   
    mnist.init()
    xtrainr, ytrainr, xtestr, ytestr = mnist.load()
    xtrainr, ytrainr, xtestr, ytestr  = getSubclass(xtrainr, ytrainr, xtestr, ytestr , labels=labels,All=All)
    return chooseRandomSubset(xtrainr,ytrainr,xtestr,ytestr,N_train=N_train,N_test=N_test,normalize=normalize,seed=seed)

def chooseRandomSubset(xtrainr_in,ytrainr,xtestr_in,ytestr,N_train=1000,N_test=1000,normalize=True,seed=0):
    if normalize:
        normf_train=norm(xtrainr_in)[:,None]
        normf_test=norm(xtestr_in)[:,None]
    else:
        normf_train=1.0
        normf_test=1.0
        
    xtrainr=xtrainr_in/normf_train
    xtestr=xtestr_in/normf_test

    #fig,ax=plt.subplots(1,1,figsize=(2,2))
    #ax.imshow(x_train_raw.reshape((-1,28,28))[0,:,:],cmap='gray')

    print('# of training images:{}'.format(np.shape(ytrainr)[0]))
    print('# of test images:{}'.format(np.shape(ytestr)[0]))
    
    n=N_train
    train_ID_sub=np.random.choice(np.arange(np.shape(ytrainr)[0]),size=n,replace=False)
    x_train=xtrainr[train_ID_sub,:]

    n=N_test
    test_ID_sub=np.random.choice(np.arange(np.shape(ytestr)[0]),size=n,replace=False)
    x_test=xtestr[test_ID_sub,:]

    fig,ax=plt.subplots(1,1,figsize=(2,2))
    ax.imshow(x_test.reshape((-1,28,28))[3,:,:],cmap='gray')
    
    ones=jnp.ones(1)[None,:]
    zeros=jnp.zeros(1)[None,:]
    y_class=blockcons(ones,zeros,10)
    dout=jnp.shape(y_class)[1]
    #fig,ax=plt.subplots(1,1,figsize=(5,3))
    #ax.imshow(y_class)

    t_train=ytrainr[train_ID_sub]
    t_test=ytestr[test_ID_sub]
    
    
    sortI=jnp.argsort(t_train)
    x_train=x_train[sortI,:]
    t_train=np.sort(t_train)
    y_train=y_class[t_train,:]
    

    sortI=jnp.argsort(t_test)
    x_test=x_test[sortI,:]
    t_test=np.sort(t_test)
    y_test=y_class[t_test,:]
    
    #samey=(y[:,None]-y[None,:])==0
    same00=(t_train[:,None]-t_train[None,:])==0
    same01=(t_train[:,None]-t_test[None,:])==0
    same11=(t_test[:,None]-t_test[None,:])==0
    return x_train,x_test,t_train,t_test,y_train,y_test,same00,same01,same11,y_class

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getCIFAR(N_train=1000,N_test=1000,normalize=True,seed=0,home='',grayscale=False,labels=[0,1],All=True):
    np.random.seed(seed)

    path_b1=home+'mypylib/CIFAR/data_batch_1'
    path_b2=home+'mypylib/CIFAR/data_batch_2'
    path_b3=home+'mypylib/CIFAR/data_batch_3'
    path_bt=home+'mypylib/CIFAR/test_batch'

    b1_dic=unpickle(path_b1)
    b2_dic=unpickle(path_b2)
    bt_dic=unpickle(path_bt)

    x_train_raw1=jnp.array(b1_dic[b'data'])
    t_train_raw1=jnp.array(b1_dic[b'labels'])
    x_train_raw2=jnp.array(b2_dic[b'data'])
    t_train_raw2=jnp.array(b2_dic[b'labels'])
    
    x_train_raw=jnp.block([[x_train_raw1],[x_train_raw2]])
    t_train_raw=jnp.concatenate([t_train_raw1,t_train_raw2])
    
    x_test_raw=jnp.array(bt_dic[b'data'])
    t_test_raw=jnp.array(bt_dic[b'labels'])

    if grayscale:
        x_train_raw=jnp.mean(x_train_raw.reshape((-1,3,32,32)),axis=1).reshape((-1,32*32))
        x_test_raw=jnp.mean(x_test_raw.reshape((-1,3,32,32)),axis=1).reshape((-1,32*32))
    
    if normalize:
        normf_train=norm(x_train_raw)[:,None]
        normf_test=norm(x_test_raw)[:,None]
    else:
        normf_train=1.0
        normf_test=1.0
    
    x_train_raw=x_train_raw/norm(x_train_raw)[:,None]
    x_test_raw=x_test_raw/norm(x_test_raw)[:,None]

    x_train_raw, t_train_raw, x_test_raw, t_test_raw  = getSubclass(x_train_raw, t_train_raw, x_test_raw, t_test_raw , labels=labels,All=All)
    
    if grayscale:
        fig,ax=plt.subplots(1,1,figsize=(2,2))
        ax.imshow(x_train_raw.reshape((-1,32,32))[0,:,:],cmap='gray')
    else:
        fig,ax=plt.subplots(1,1,figsize=(2,2))
        ax.imshow(x_train_raw.reshape((-1,3,32,32))[0,:,:,:].transpose((1,2,0)))#,cmap='gray')

    n=N_train
    train_ID_sub=np.random.choice(np.arange(np.shape(t_train_raw)[0]),size=n,replace=False)
    x_train=x_train_raw[train_ID_sub,:]

    n=N_test
    test_ID_sub=np.random.choice(np.arange(np.shape(t_test_raw)[0]),size=n,replace=False)
    x_test=x_test_raw[test_ID_sub,:]
    
    #####
    ones=jnp.ones(1)[None,:]
    zeros=jnp.zeros(1)[None,:]
    y_class=blockcons(ones,zeros,10)
    dout=jnp.shape(y_class)[1]
    #fig,ax=plt.subplots(1,1,figsize=(5,3))
    #ax.imshow(y_class)

    t_train=t_train_raw[train_ID_sub]
    t_test=t_test_raw[test_ID_sub]

    sortI=jnp.argsort(t_train)
    x_train=x_train[sortI,:]
    t_train=t_train[sortI]
    y_train=y_class[t_train,:]
    
    sortI=jnp.argsort(t_test)
    x_test=x_test[sortI,:]
    t_test=t_test[sortI]
    y_test=y_class[t_test,:]
    
    #samey=(y[:,None]-y[None,:])==0
    same00=(t_train[:,None]-t_train[None,:])==0
    same01=(t_train[:,None]-t_test[None,:])==0
    same11=(t_test[:,None]-t_test[None,:])==0
    return x_train,x_test,t_train,t_test,y_train,y_test,same00,same01,same11,y_class

def norm(x):
    #x= rows: samples, columns: features
    return jnp.sqrt(jnp.sum(jnp.square(x),axis=1))
def getTau(Pa):
    return jnp.sqrt(2.0)*jsc.special.erfinv(1.0-2.0*Pa)
def getPa(tau):
    return 0.5*(1.0-jsc.special.erf(tau/np.sqrt(2.0)))
def gauss(x):
    return 1.0/jnp.sqrt(2.0*jnp.pi)*jnp.exp(-0.5*jnp.square(x))
def recMu(tau,sigma):
    return sigma*(gauss(tau)-tau*getPa(tau))
def I2(thetar,taur,n=100):
    theta=thetar[None,:,None]
    xi=(jnp.pi-thetar)/2.0
    tau=taur[:,None,None]
    g=1.0
    phi=jnp.linspace(0,xi,n).T[None,:,:]
    L_term1=2*jnp.sin(phi+theta)*jnp.sin(phi)*jnp.exp(-0.5*jnp.square(g*tau/jnp.sin(phi)))
    L_term2=g*tau*(jnp.sin(phi+theta)+jnp.sin(phi))*jnp.sqrt(jnp.pi/2)*jsc.special.erf(jnp.sqrt(0.5)*g*tau/jnp.sin(phi))
    inte=jnp.trapz(L_term1+L_term2,x=phi,axis=-1)
    return inte#,L_term1+L_term2,phi
#def g(tau): return 1.0/jnp.sqrt(2.0*jnp.pi)*jnp.exp(-jnp.square(tau)/2)
def prednKb(gamma,tau,n=1000,chunk_size=10000): 
    #thetar=jnp.array([jnp.arccos(gamma)])
    thetar=jnp.arccos(gamma)    
    v=2*I2(thetar,jnp.array([tau]),n=n)-tau*jnp.sqrt(2*jnp.pi)*(1.0+jnp.cos(thetar))
    #theta=jnp.array([jnp.arccos(gamma)]) #current
    #v=I2_chunk(theta.flatten(),jnp.array([tau]),n=n,chunk_size=chunk_size,silence=True)
    #v=jnp.reshape(v,jnp.shape(theta))
    return jnp.squeeze(v)#/(2*np.pi)#*1/((tau**2+1)*getPa(tau)-tau*g(tau))
def lookupy(D,x=None,y=None):
    idx=jnp.digitize(D,x)
    return x[idx],y[idx]

def getConverter_basic(tau=2.0,resolution=10000,trapz_n=10000,chunk_size=10000):
    covars=jnp.linspace(-1,1,resolution)
    a=np.zeros((np.shape(covars)[0]))
    for i,covar in enumerate(covars):
        a[i]=prednKb(jnp.array([covar]),tau,n=trapz_n,chunk_size=chunk_size)
        print('\r{:4f}'.format(i/resolution),end='')
    a=jnp.array(a)
    def converter(gamma,getTau=False):
        if getTau:
            return tau
        else:
            _,Kh_norm=lookupy(gamma,x=covars,y=a)
            return Kh_norm
    return converter

def getConverter_chunk(tau=2.0,resolution=10000,trapz_n=10000,chunk_size=10000,silence=False):
    covars=jnp.linspace(-1,1,resolution)
    N=resolution
    
    n_chunks=int(jnp.ceil(N/chunk_size))
    if N != chunk_size:
        remain=N%chunk_size
        if remain==0:
            pad_size=0
        else:
            pad_size=chunk_size-N%chunk_size
        covarss=jnp.reshape(jnp.concatenate([covars,jnp.zeros(pad_size)]),(n_chunks,chunk_size))
        print('chopped')
    else:
        covarss=np.array([covars])
        print('passed')
    inte=[]
    for i,covars_c in enumerate(covarss):
        ival=prednKb(covars_c,tau,n=trapz_n,chunk_size=chunk_size)
        #print('\r{:4f}'.format(i/resolution),end='')
        inte.append(jnp.squeeze(ival))
        if not silence:
            print('\rProgress={:.2f}%, #chunks:{}'.format(100*i/n_chunks,n_chunks),end='')
    a=jnp.concatenate(inte)[:N]
    a=jnp.array(a)
    def converter(gamma,getTau=False):
        if getTau:
            return tau
        else:
            _,Kh_norm=lookupy(gamma,x=covars,y=a)
            return Kh_norm     
    return converter

def getConverter_save_load(tau=2.0,resolution=10000,trapz_n=10000,chunk_size=10000,home=''):
    name='srGP_tau_{}_res_{}_trapz_{}'.format(tau,resolution,trapz_n)
    path_name=home+'mypylib/lookup/'+name
    covars=jnp.linspace(-1,1,resolution)
    if exists(path_name+'.npy'):
        a=np.load(path_name+'.npy')
        print('loaded from a file')
    else:
        a=np.zeros((np.shape(covars)[0]))
        for i,covar in enumerate(covars):
            a[i]=prednKb(covar,tau,n=trapz_n,chunk_size=chunk_size)
            print('\r{:4f}'.format(i/resolution),end='')
        np.save(path_name,a)
    a=jnp.array(a)
    def converter(gamma,getTau=False):
        if getTau:
            return tau
        else:
            _,Kh_norm=lookupy(gamma,x=covars,y=a)
            return Kh_norm
    return converter


def getConverter_chunk_save_load(tau=2.0,resolution=10000,trapz_n=10000,chunk_size=10000,silence=False,home=''):
    name='srGP_tau_{}_res_{}_trapz_{}'.format(tau,resolution,trapz_n)
    path_name=home+'mypylib/lookup/'+name
    covars=jnp.linspace(-1,1,resolution)
    if exists(path_name+'.npy'):
        a=np.load(path_name+'.npy')
        print('loaded from a file')
    else:

        N=resolution
        n_chunks=int(jnp.ceil(N/chunk_size))
        if N != chunk_size:
            remain=N%chunk_size
            if remain==0:
                pad_size=0
            else:
                pad_size=chunk_size-N%chunk_size
            covarss=jnp.reshape(jnp.concatenate([covars,jnp.zeros(pad_size)]),(n_chunks,chunk_size))
            print('chopped')
        else:
            covarss=np.array([covars])
            print('passed')
        inte=[]
        for i,covars_c in enumerate(covarss):
            ival=prednKb(covars_c,tau,n=trapz_n,chunk_size=chunk_size)
            #print('\r{:4f}'.format(i/resolution),end='')
            inte.append(jnp.squeeze(ival))
            if not silence:
                print('\rProgress={:.2f}%, #chunks:{}'.format(100*i/n_chunks,n_chunks),end='')
        a=np.concatenate(inte)[:N]
        np.save(path_name,a)
    a=jnp.array(a)
    def converter(gamma,getTau=False):
        if getTau:
            return tau
        else:
            _,Kh_norm=lookupy(gamma,x=covars,y=a)
            return Kh_norm     
    return converter


def block_inv(A,B,N):
    ABiA=jnp.matmul(jnp.matmul(A,np.linalg.inv(B)),A)
    Fi=-(ABiA+(N-2)*A-(N-1)*B)
    preinv=A-B
    F=jnp.linalg.inv(Fi)
    G=jnp.linalg.inv(preinv)+F
    #F=jnp.array(solve(Fi,np.eye(jnp.shape(Fi)[0])))
    #G=jnp.array(solve(preinv,np.eye(jnp.shape(preinv)[0])))+F
    return G,F
def blockmul(A,B,C,D,N):
    F=jnp.matmul(A,D)+jnp.matmul(B,C)+(N-2)*jnp.matmul(B,D)
    G=jnp.matmul(A,C)+(N-1)*jnp.matmul(B,D)
    return G,F
def blockcons(G,F,N):
    R=[]
    for i in range(N):
        Ri=[]
        for j in range(N):
            if i==j:
                Rij=jnp.array(G)*1.0
            else:
                Rij=jnp.array(F)*1.0
            Ri.append(Rij)
        R.append(Ri)
    return jnp.block(R)

#########

def getosig(tau,m=0):
    I0=I2(np.array([0]),np.array([tau]),n=1000)
    if m==0:
        osig=np.sqrt(np.pi/(I0-tau*np.sqrt(2*np.pi)))
    else:
        Ih=I2(np.array([np.pi/2]),np.array([tau]),n=1000)
        osig=np.sqrt(2*np.pi/(2*I0-2*tau*np.sqrt(2*np.pi)+m*(2*Ih-tau*np.sqrt(2*np.pi)) ))
    return osig

def effD(sv):
    return jnp.square(jnp.sum(sv))/jnp.sum(jnp.square(sv))

def f(i,j,k,l,mw=4,mh=10): zw=mw+1;zh=mh+1; return (zh-jnp.min(jnp.array([jnp.abs(i-k),zh])))*(zw-jnp.min(jnp.array([jnp.abs(j-l),zw])))
def zp(X,p=0): return jnp.block([[jnp.zeros((p,jnp.shape(X)[1]))],[X[p:,:]]])

def getTile(N=10,d=0,j=0,mh=10,mw=4):
    zh=mh+1
    zw=mw+1
    A=np.zeros((2*zh,N))
    A[d:(d+zh), np.max([int(j-(mw/2)),0]):np.min([int(j+(mw/2)+1),N])]=1.0
    return A

def getF(N=10,d=0,j=0,l=1,mh=10,mw=4):
    A0=getTile(N=N,d=0,j=j,mh=mh,mw=mw)
    A=getTile(N=N,d=d,j=l,mh=mh,mw=mw)
    return np.sum(A*A0)
def zpA(X,p=0): return jnp.block([[X[p:,:]],[jnp.zeros((p,jnp.shape(X)[1]))]])
def zpB(X,p=0): return jnp.block([[jnp.zeros((p,jnp.shape(X)[1]))],[X[:-p,:]]])
def getKab1(X0,X1,sigma=1.2,mh=4,mw=2,id=0,kd=0):
    #X should be N x Ns
    N=jnp.shape(X1)[0]
    Ns=jnp.shape(X1)[1]
    D=jnp.zeros((N,Ns))
    for q in range(-mw,mw+1):
        vpa=jnp.array([getF(N=N,d=np.abs(kd-id),j=j,l=j+q,mw=mw,mh=mh) for j in range(N)])
        if q>=0:
            D+=(vpa[:,None]*zpA(X1,p=np.abs(q)))
        else:
            D+=(vpa[:,None]*zpB(X1,p=np.abs(q)))
    M=jnp.matmul(X0.T,D)
    K1=((sigma**2)/((mh+1)*(mw+1)))*(1/N)*M#+M.T)
    return K1

def accuracy(y_pred,y):
    return jnp.mean(y_pred == y)

def postCov(K00,K01,K11,s0=0.0):
    return K11-jnp.matmul(jnp.matmul(K01.T,jnp.linalg.inv(K00+(s0**2)*jnp.eye(np.shape(K00)[0]))),K01)

def train_multi(A_raw,B,C,D,y_train,corr=True,s0=0.0):
    dout=np.shape(y_train)[1]
    #uncert=(s0**2)*jnp.eye(np.shape(A_raw)[0])
    uncert=s0*jnp.eye(np.shape(A_raw)[0])
    A=A_raw+uncert
    if corr:
        Gr,Fr=block_inv(A,B,dout)
    else:
        Gr=jnp.array(jnp.linalg.inv(A))
        #Gr=jnp.array(solve(A,np.eye(jnp.shape(A)[0])))
        Fr=jnp.array(B)
    G,F=blockmul(Gr,Fr,C,D,dout)
    
    u_r=jnp.mean(y_train)*0.0#,axis=0)
    y_train=jnp.array(y_train-u_r)*1.0
    yr_test=[]
    for i in range(dout):
        yoi=jnp.squeeze(y_train[:,i])
        sy=jnp.squeeze(jnp.sum(y_train[:,jnp.arange(dout)!=i],axis=1))
        y_test_i=jnp.matmul(yoi[None,:],G)+jnp.matmul(sy[None,:],F)
        yr_test.append(jnp.squeeze(y_test_i))
    return jnp.array(yr_test).T+u_r

def train_single_fast(A_raw,C,y_train,s0=0.0):
    dout=np.shape(y_train)[1]
    #uncert=(s0**2)*jnp.eye(np.shape(A_raw)[0])
    uncert=s0*jnp.eye(np.shape(A_raw)[0])
    A=A_raw+uncert
    lower=True
    L = jnp.linalg.cholesky(A)
    yr_test=[]
    for i in range(dout):
        alpha = jsc.linalg.cho_solve((L, lower), y_train[:,i])
        y_test_i=jnp.matmul(alpha.T,C)
        yr_test.append(jnp.squeeze(y_test_i))
    return jnp.array(yr_test).T

def getNN(yr_test,y_class):
    pdist=pairwise_distances(yr_test, y_class, metric='euclidean')
    tp_test=jnp.argmin(pdist,axis=1)
    return y_class[tp_test,:],tp_test

def getMax(yr_test):
    tp_test=jnp.argmax(yr_test,axis=1)
    return tp_test

def I2(thetar,taur,n=100):
    theta=thetar[None,:,None]
    xi=(jnp.pi-thetar)/2.0
    tau=taur[:,None,None]
    g=1.0
    phi=jnp.linspace(0,xi,n).T[None,:,:]
    L_term1=2*jnp.sin(phi+theta)*jnp.sin(phi)*jnp.exp(-0.5*jnp.square(g*tau/jnp.sin(phi)))
    L_term2=g*tau*(jnp.sin(phi+theta)+jnp.sin(phi))*jnp.sqrt(jnp.pi/2)*jsc.special.erf(jnp.sqrt(0.5)*g*tau/jnp.sin(phi))
    inte=jnp.trapz(L_term1+L_term2,x=phi,axis=-1)
    return inte

def I2_chunk(theta,tau,n=1000,chunk_size=10000,silence=False):
    N=jnp.shape(theta)[0]
    n_chunks=int(jnp.ceil(N/chunk_size))
    if N != chunk_size:
        remain=N%chunk_size
        if remain==0:
            pad_size=0
        else:
            pad_size=chunk_size-N%chunk_size
        thetas=jnp.reshape(jnp.concatenate([theta,jnp.zeros(pad_size)]),(n_chunks,chunk_size))
    else:
        #thetas=jnp.array([theta])
        thetas=theta*1.0
        
    inte=[]
    for i,theta_c in enumerate(thetas):
        ival=I2(theta_c,tau,n=n)
        inte.append(jnp.squeeze(ival))
        if not silence:
            print('\rProgress={:.2f}%, #chunks:{}'.format(100*i/n_chunks,n_chunks),end='')
    inte=jnp.concatenate(inte)[:N]
    #print('')
    return inte

def getGamma(K01,sqrtK00K11):
    proj=K01/sqrtK00K11
    proj=jnp.where(proj>1.0,1.0,proj)
    proj=jnp.where(proj<-1.0,-1.0,proj)
    return proj

def arcK(K01,sqrtK00K11):
    proj=K01/sqrtK00K11
    proj=jnp.where(proj>1.0,1.0,proj)
    proj=jnp.where(proj<-1.0,-1.0,proj)
    return jnp.arccos(proj)

#If L=2, K_01sL will have two elements. The first hidden layer preactivation cov. and one after that.
#In the main code, use Ls=[1,2,3,...] (start from 1)
#When performing a deep Hebbian learning, ignore the last element of K_01sL.
def getSCK_old(x0,x1,mw0=0,mh0=0,mw=4,mh=10,sigma=1.0,taus=2.0,L=2,getT=False,ntrapz=1000,chunk_size=100000,layer_zero_mean=False,input_unit_ball=True):
    if layer_zero_mean:
        x0=x0-jnp.mean(x0,axis=1)[:,None]
        x1=x1-jnp.mean(x1,axis=1)[:,None]
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))[:,None]
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))[:,None]
    if np.shape(taus)==():
        taus=np.ones(L-1)*taus
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]
    #initialize
    K_01s=[]
    zw=mw+1
    zh=mh+1
    for kd in range(mh0+1+1):
        K_01s.append(getKab1(x0.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd))
    K_00s=[]
    for kd in range(mh0+1+1):
        K_00s.append(jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_11s=[]
    for kd in range(mh0+1+1):
        K_11s.append(jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_01sL=[]
    P_01sL=[]
    K_01sL.append(jnp.array(np.copy(K_01s[0])))
    TT=[];  
    if getT:
        TT.append(arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:])))
    for i in range(L-1):
        tau=taus[i]
        sqrtK00K11=jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]) #current
        if layer_zero_mean:
            offset_01=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
            offset_00=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_00s[0]))[None,:]
            offset_11=recMu(tau,jnp.sqrt(K_11s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
        else:
            offset_01=0.0
            offset_00=0.0
            offset_11=0.0
        simPhi_01s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_01s)):
            theta=arcK(K_01s[kd],sqrtK00K11) #current
            I=I2_chunk(theta.flatten(),jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            I=jnp.reshape(I,jnp.shape(theta))
            simPhi_01_temp=1.0/(2.0*jnp.pi)*sqrtK00K11*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current
            simPhi_01s.append(simPhi_01_temp)
        simPhi_01=jnp.array(simPhi_01s[0])*1.0
        K_01s=[]
        for kd in range(zh+1):
            S=jnp.zeros((s0,s1))
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_01s)-1))
                S+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_01s[id]-offset_01)
            K_01_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_01+2*((sigma**2)/(zh*zw))*S #next
            K_01s.append(jnp.array(np.copy(K_01_temp))) #next
        K_01sL.append(jnp.array(np.copy(K_01s[0])))

        ###for K00s and K11s
        simPhi_00s=[]
        simPhi_11s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_00s)):
            theta=arcK(K_00s[kd],K_00s[0])
            I=I2_chunk(theta,jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            simPhi_00_temp=1.0/(2.0*jnp.pi)*K_00s[0]*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current

            theta=arcK(K_11s[kd],K_11s[0])
            I=I2_chunk(theta,jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            simPhi_11_temp=1.0/(2.0*jnp.pi)*K_11s[0]*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current
            #print(simPhi_11_temp-jnp.diag(simPhi_01s[kd]))
            simPhi_00s.append(simPhi_00_temp)
            simPhi_11s.append(simPhi_11_temp)
        simPhi_00=jnp.array(simPhi_00s[0])*1.0
        simPhi_11=jnp.array(simPhi_11s[0])*1.0

        K_00s=[]
        K_11s=[]
        for kd in range(zh+1):
            S00=jnp.zeros(s0)
            S11=jnp.zeros(s1)
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_00s)-1))
                S00+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_00s[id]-offset_00)
                S11+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_11s[id]-offset_11)
            K_00_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_00+2*((sigma**2)/(zh*zw))*S00 #next
            K_11_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_11+2*((sigma**2)/(zh*zw))*S11 #next
            #print(K_11_temp-jnp.diag(K_01s[kd]))
            K_00s.append(jnp.array(np.copy(K_00_temp)))
            K_11s.append(jnp.array(np.copy(K_11_temp)))
        if getT:
            TT.append(arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:])))

    return K_01s[0],K_01sL,TT

#SK_L: K similarity starting from the readouts from first hidden layer, to the readouts from the last hidden layer. If L=2, len(SK_L)==1. L-1 is the number of hidden layers. L+2 is the number of total layers.
#TT: Hidden similarity starting from the first hidden layer. However, there is an extra (readout output).
def getSCK(x0,x1,mw0=0,mh0=0,mw=0,mh=0,sigma=1.0,taus=2.0,L=2,getT=False,ntrapz=1000,chunk_size=100000,layer_zero_mean=False,input_unit_ball=True):
    if layer_zero_mean:
        x0=x0-jnp.mean(x0,axis=1)[:,None]
        x1=x1-jnp.mean(x1,axis=1)[:,None]
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))[:,None]
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))[:,None]
    if np.shape(taus)==():
        taus=np.ones(L-1)*taus
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]
    #initialize
    K_01s=[]
    zw=mw+1
    zh=mh+1
    for kd in range(mh0+1+1):
        K_01s.append(getKab1(x0.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd))
    K_00s=[]
    for kd in range(mh0+1+1):
        K_00s.append(jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_11s=[]
    for kd in range(mh0+1+1):
        K_11s.append(jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    #P_01sL=[]
    SK_L=[]
    TT=[];  
    if getT:
        TT.append(arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:])))
    for i in range(L-1):
        tau=taus[i]
        sqrtK00K11=jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]) #current
        if layer_zero_mean:
            offset_01=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
            offset_00=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_00s[0]))[None,:]
            offset_11=recMu(tau,jnp.sqrt(K_11s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
        else:
            offset_01=0.0
            offset_00=0.0
            offset_11=0.0
        simPhi_01s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_01s)):
            theta=arcK(K_01s[kd],sqrtK00K11) #current
            I=I2_chunk(theta.flatten(),jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            I=jnp.reshape(I,jnp.shape(theta))
            simPhi_01_temp=1.0/(2.0*jnp.pi)*sqrtK00K11*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current
            simPhi_01s.append(simPhi_01_temp)
        simPhi_01=jnp.array(simPhi_01s[0])*1.0
        
        K_01s=[]
        for kd in range(zh+1):
            S=jnp.zeros((s0,s1))
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_01s)-1))
                S+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_01s[id]-offset_01)
            K_01_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_01+2*((sigma**2)/(zh*zw))*S #next
            K_01s.append(jnp.array(np.copy(K_01_temp))) #next
        SK_L.append(jnp.array(np.copy(K_01s[0])))

        ###for K00s and K11s
        simPhi_00s=[]
        simPhi_11s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_00s)):
            theta=arcK(K_00s[kd],K_00s[0])
            I=I2_chunk(theta,jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            simPhi_00_temp=1.0/(2.0*jnp.pi)*K_00s[0]*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current

            theta=arcK(K_11s[kd],K_11s[0])
            I=I2_chunk(theta,jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            simPhi_11_temp=1.0/(2.0*jnp.pi)*K_11s[0]*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current
            #print(simPhi_11_temp-jnp.diag(simPhi_01s[kd]))
            simPhi_00s.append(simPhi_00_temp)
            simPhi_11s.append(simPhi_11_temp)
        simPhi_00=jnp.array(simPhi_00s[0])*1.0
        simPhi_11=jnp.array(simPhi_11s[0])*1.0

        K_00s=[]
        K_11s=[]
        for kd in range(zh+1):
            S00=jnp.zeros(s0)
            S11=jnp.zeros(s1)
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_00s)-1))
                S00+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_00s[id]-offset_00)
                S11+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_11s[id]-offset_11)
            K_00_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_00+2*((sigma**2)/(zh*zw))*S00 #next
            K_11_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_11+2*((sigma**2)/(zh*zw))*S11 #next
            #print(K_11_temp-jnp.diag(K_01s[kd]))
            K_00s.append(jnp.array(np.copy(K_00_temp)))
            K_11s.append(jnp.array(np.copy(K_11_temp)))
        if getT:
            TT.append(arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:])))

    return SK_L,TT

def getSCK_fast(x0,x1,convert,mw0=0,mh0=0,mw=4,mh=10,sigma=1.0,L=2,getT=False,layer_zero_mean=False,input_unit_ball=True):
    if layer_zero_mean:
        x0=x0-jnp.mean(x0,axis=1)[:,None]
        x1=x1-jnp.mean(x1,axis=1)[:,None]
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))[:,None]
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))[:,None]
        
    tau=convert(None,getTau=True)
    
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]
    #initialize
    K_01s=[]
    zw=mw+1
    zh=mh+1
    for kd in range(mh0+1+1):
        K_01s.append(getKab1(x0.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd))
    K_00s=[]
    for kd in range(mh0+1+1):
        K_00s.append(jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_11s=[]
    for kd in range(mh0+1+1):
        K_11s.append(jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_01sL=[]
    P_01sL=[]
    TT=[]; 
    if getT:
        TT.append(arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:])))
    for i in range(L-1):
        sqrtK00K11=jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]) #current
        if layer_zero_mean:
            offset_01=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
            offset_00=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_00s[0]))[None,:]
            offset_11=recMu(tau,jnp.sqrt(K_11s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
        else:
            offset_01=0.0
            offset_00=0.0
            offset_11=0.0
        simPhi_01s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_01s)):
            gamma=getGamma(K_01s[kd],sqrtK00K11) #current
            simPhi_01_temp=1.0/(2.0*jnp.pi)*sqrtK00K11*convert(gamma)
            simPhi_01s.append(simPhi_01_temp)
        simPhi_01=jnp.array(simPhi_01s[0])*1.0
        K_01s=[]
        for kd in range(zh+1):
            S=jnp.zeros((s0,s1))
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_01s)-1))
                S+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_01s[id]-offset_01)
                
            K_01_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_01+2*((sigma**2)/(zh*zw))*S #next
            K_01s.append(jnp.array(np.copy(K_01_temp))) #next
        K_01sL.append(jnp.array(np.copy(K_01s[0])))

        ###for K00s and K11s
        simPhi_00s=[]
        simPhi_11s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_00s)):
            gamma=getGamma(K_00s[kd],K_00s[0])
            simPhi_00_temp=1.0/(2.0*jnp.pi)*K_00s[0]*convert(gamma) #current

            gamma=getGamma(K_11s[kd],K_11s[0])
            simPhi_11_temp=1.0/(2.0*jnp.pi)*K_11s[0]*convert(gamma) #current
            #print(simPhi_11_temp-jnp.diag(simPhi_01s[kd]))
            simPhi_00s.append(simPhi_00_temp)
            simPhi_11s.append(simPhi_11_temp)
        simPhi_00=jnp.array(simPhi_00s[0])*1.0
        simPhi_11=jnp.array(simPhi_11s[0])*1.0

        K_00s=[]
        K_11s=[]
        for kd in range(zh+1):
            S00=jnp.zeros(s0)
            S11=jnp.zeros(s1)
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_00s)-1))
                S00+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_00s[id]-offset_00)
                S11+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_11s[id]-offset_11)
            K_00_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_00+2*((sigma**2)/(zh*zw))*S00 #next
            K_11_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_11+2*((sigma**2)/(zh*zw))*S11 #next
            #print(K_11_temp-jnp.diag(K_01s[kd]))
            K_00s.append(jnp.array(np.copy(K_00_temp)))
            K_11s.append(jnp.array(np.copy(K_11_temp)))
        if getT:
            TT.append(arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:])))

    return K_01sL,TT


def getSCK_fast_and_save(x0,x1,convert,mw0=0,mh0=0,mw=4,mh=10,sigma=1.0,L=2,layer_zero_mean=False,input_unit_ball=True,home='',label=''):
    name='temp_GP_Kernel_'+label
    path_name=home+'mypylib/lookup/'+name
    load_names=[]
    
    if layer_zero_mean:
        x0=x0-jnp.mean(x0,axis=1)[:,None]
        x1=x1-jnp.mean(x1,axis=1)[:,None]
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))[:,None]
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))[:,None]
        
    tau=convert(None,getTau=True)
    
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]
    #initialize
    K_01s=[]
    zw=mw+1
    zh=mh+1
    for kd in range(mh0+1+1):
        K_01s.append(getKab1(x0.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd))
    K_00s=[]
    for kd in range(mh0+1+1):
        K_00s.append(jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_11s=[]
    for kd in range(mh0+1+1):
        K_11s.append(jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    for i in range(L-1):
        sqrtK00K11=jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]) #current
        if layer_zero_mean:
            offset_01=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
            offset_00=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_00s[0]))[None,:]
            offset_11=recMu(tau,jnp.sqrt(K_11s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
        else:
            offset_01=0.0
            offset_00=0.0
            offset_11=0.0
        simPhi_01s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_01s)):
            gamma=getGamma(K_01s[kd],sqrtK00K11) #current
            simPhi_01_temp=1.0/(2.0*jnp.pi)*sqrtK00K11*convert(gamma)
            simPhi_01s.append(simPhi_01_temp)
        simPhi_01=jnp.array(simPhi_01s[0])*1.0
                     
        np.save(path_name+'_L{}'.format(i),simPhi_01)
        print('{}th layer kernel saved'.format(i))
        load_names.append(path_name+'_L{}'.format(i)+'.npy')
                     
        K_01s=[]
        for kd in range(zh+1):
            S=jnp.zeros((s0,s1))
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_01s)-1))
                S+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_01s[id]-offset_01)
            K_01_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_01+2*((sigma**2)/(zh*zw))*S #next
            #K_01s.append(jnp.array(np.copy(K_01_temp))) #next
            K_01s.append(K_01_temp)

        #np.save(path_name+'_L{}'.format(i),K_01s[0])
        #print('{}th layer kernel saved'.format(i))
        #load_names.append(path_name+'_L{}'.format(i)+'.npy')
        
        ###for K00s and K11s
        simPhi_00s=[]
        simPhi_11s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_00s)):
            gamma=getGamma(K_00s[kd],K_00s[0])
            simPhi_00_temp=1.0/(2.0*jnp.pi)*K_00s[0]*convert(gamma) #current

            gamma=getGamma(K_11s[kd],K_11s[0])
            simPhi_11_temp=1.0/(2.0*jnp.pi)*K_11s[0]*convert(gamma) #current
            #print(simPhi_11_temp-jnp.diag(simPhi_01s[kd]))
            simPhi_00s.append(simPhi_00_temp)
            simPhi_11s.append(simPhi_11_temp)
        simPhi_00=jnp.array(simPhi_00s[0])*1.0
        simPhi_11=jnp.array(simPhi_11s[0])*1.0

        K_00s=[]
        K_11s=[]
        for kd in range(zh+1):
            S00=jnp.zeros(s0)
            S11=jnp.zeros(s1)
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_00s)-1))
                S00+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_00s[id]-offset_00)
                S11+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_11s[id]-offset_11)
            K_00_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_00+2*((sigma**2)/(zh*zw))*S00 #next
            K_11_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_11+2*((sigma**2)/(zh*zw))*S11 #next
            #print(K_11_temp-jnp.diag(K_01s[kd]))
            #K_00s.append(jnp.array(np.copy(K_00_temp)))
            #K_11s.append(jnp.array(np.copy(K_11_temp)))
            K_00s.append(K_00_temp)
            K_11s.append(K_11_temp)
    print('done')
    return load_names
    #return K_01s[0],arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]))
    
####### Same as 'getSCK_fast_and_save', but training the last layer with 0.5 sparsity.
def getSCK_last_layer_fast_and_save(x0,x1,convertA,convertL,mw0=0,mh0=0,mw=4,mh=10,sigma=1.0,L=2,input_unit_ball=True,home='',label=''):
    name='temp_GP_Kernel_'+label
    path_name=home+'mypylib/lookup/'+name
    load_names=[]
    
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))[:,None]
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))[:,None]
            
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]
    #initialize
    K_01s=[]
    zw=mw+1
    zh=mh+1
    for kd in range(mh0+1+1):
        K_01s.append(getKab1(x0.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd))
    K_00s=[]
    for kd in range(mh0+1+1):
        K_00s.append(jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_11s=[]
    for kd in range(mh0+1+1):
        K_11s.append(jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    for i in range(L-1):
        if i==L-2:
            convert=convertL
        else:
            convert=convertA
        sqrtK00K11=jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]) #current
        offset_01=0.0
        offset_00=0.0
        offset_11=0.0
        simPhi_01s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_01s)):
            gamma=getGamma(K_01s[kd],sqrtK00K11) #current
            simPhi_01_temp=1.0/(2.0*jnp.pi)*sqrtK00K11*convert(gamma)
            simPhi_01s.append(simPhi_01_temp)
        simPhi_01=jnp.array(simPhi_01s[0])*1.0
                     
        np.save(path_name+'_L{}'.format(i),simPhi_01)
        print('{}th layer kernel saved'.format(i))
        load_names.append(path_name+'_L{}'.format(i)+'.npy')
                     
        K_01s=[]
        for kd in range(zh+1):
            S=jnp.zeros((s0,s1))
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_01s)-1))
                S+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_01s[id]-offset_01)
            K_01_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_01+2*((sigma**2)/(zh*zw))*S #next
            #K_01s.append(jnp.array(np.copy(K_01_temp))) #next
            K_01s.append(K_01_temp)

        #np.save(path_name+'_L{}'.format(i),K_01s[0])
        #print('{}th layer kernel saved'.format(i))
        #load_names.append(path_name+'_L{}'.format(i)+'.npy')
        
        ###for K00s and K11s
        simPhi_00s=[]
        simPhi_11s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_00s)):
            gamma=getGamma(K_00s[kd],K_00s[0])
            simPhi_00_temp=1.0/(2.0*jnp.pi)*K_00s[0]*convert(gamma) #current

            gamma=getGamma(K_11s[kd],K_11s[0])
            simPhi_11_temp=1.0/(2.0*jnp.pi)*K_11s[0]*convert(gamma) #current
            #print(simPhi_11_temp-jnp.diag(simPhi_01s[kd]))
            simPhi_00s.append(simPhi_00_temp)
            simPhi_11s.append(simPhi_11_temp)
        simPhi_00=jnp.array(simPhi_00s[0])*1.0
        simPhi_11=jnp.array(simPhi_11s[0])*1.0

        K_00s=[]
        K_11s=[]
        for kd in range(zh+1):
            S00=jnp.zeros(s0)
            S11=jnp.zeros(s1)
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_00s)-1))
                S00+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_00s[id]-offset_00)
                S11+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_11s[id]-offset_11)
            K_00_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_00+2*((sigma**2)/(zh*zw))*S00 #next
            K_11_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_11+2*((sigma**2)/(zh*zw))*S11 #next
            #print(K_11_temp-jnp.diag(K_01s[kd]))
            #K_00s.append(jnp.array(np.copy(K_00_temp)))
            #K_11s.append(jnp.array(np.copy(K_11_temp)))
            K_00s.append(K_00_temp)
            K_11s.append(K_11_temp)
    print('done')
    return load_names
    #return K_01s[0],arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]))

#### Same as getSCK_fast_and_save, but optimized for flat correlation. Proper last layer

def getSCK_flat_fast_and_save(x0,x1,convert,mw=4,sigma=1.0,L=2,input_unit_ball=True,home='',label='',saveKh=False):
    name='temp_GP_Kernel_'+label
    path_name=home+'mypylib/lookup/'+name
    load_names=[]
    
    if saveKh:
        name2='temp_GP_Kernel_h_'+label
        path_name2=home+'mypylib/lookup/'+name2
        load_names2=[]
        
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))[:,None]
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))[:,None]
        
    tau=convert(None,getTau=True)
    
    int0=convert(0)
    int1=convert(1)
    
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]
    #initialize
    zw=mw+1
    K_01=getKab1(x0.T,x1.T,sigma=sigma,mw=0,mh=0,id=0,kd=0)
    K_00=jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=0,mh=0,id=0,kd=0))
    K_11=jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=0,mh=0,id=0,kd=0))
    for i in range(L-1):
        if saveKh:
            np.save(path_name2+'_L{}'.format(i),K_01)
            load_names2.append(path_name2+'_L{}'.format(i)+'.npy')

        sqrtK00K11=jnp.sqrt(K_00[:,None]*K_11[None,:]) #current

        gamma=getGamma(K_01,sqrtK00K11) #current
        simPhi_01=1.0/(2.0*jnp.pi)*sqrtK00K11*convert(gamma)
        
        np.save(path_name+'_L{}'.format(i),simPhi_01)
        print('{}th layer kernel saved'.format(i))
        load_names.append(path_name+'_L{}'.format(i)+'.npy')
        
        Eg_01= 1.0/(2.0*jnp.pi)*sqrtK00K11* int0
        K_01=(sigma**2)* (simPhi_01+ mw* Eg_01)
        
        ###for K00s and K11s
        simPhi_00=1.0/(2.0*jnp.pi)*K_00*int1 #current
        simPhi_11=1.0/(2.0*jnp.pi)*K_11*int1 #current
        
        Eg_00=1.0/(2.0*jnp.pi)*K_00* int0
        Eg_11=1.0/(2.0*jnp.pi)*K_11* int0
        K_00=(sigma**2)*(simPhi_00 + mw*Eg_00)
        K_11=(sigma**2)*(simPhi_11 + mw*Eg_11)
            
    print('done')
    if saveKh:
        return load_names,load_names2
    else:
        return load_names
    #return K_01s[0],arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]))

def getarccos_fast_and_save(x0,x1,convert,sigma=1.0,sigmab=0.0,L=2,layer_zero_mean=False,input_unit_ball=True,home='',label=''):
    name='temp_GP_Kernel_'+label
    path_name=home+'mypylib/lookup/'+name
    load_names=[]
    
    if layer_zero_mean:
        x0=x0-jnp.mean(x0,axis=1)[:,None]
        x1=x1-jnp.mean(x1,axis=1)[:,None]
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))[:,None]
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))[:,None]
     
    mw0=0
    mh0=0
    mw=0
    mh=0
    
    #tau=convert(None,getTau=True)
    tau=1e-4
    
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]
    #initialize
    K_01s=[]
    zw=mw+1
    zh=mh+1
    for kd in range(mh0+1+1):
        K_01s.append(getKab1(x0.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd))
    K_00s=[]
    for kd in range(mh0+1+1):
        K_00s.append(jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_11s=[]
    for kd in range(mh0+1+1):
        K_11s.append(jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    for i in range(L-1):
        sqrtK00K11=jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]) #current
        if layer_zero_mean:
            offset_01=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
            offset_00=recMu(tau,jnp.sqrt(K_00s[0]))[:,None]*recMu(tau,jnp.sqrt(K_00s[0]))[None,:]
            offset_11=recMu(tau,jnp.sqrt(K_11s[0]))[:,None]*recMu(tau,jnp.sqrt(K_11s[0]))[None,:]
        else:
            offset_01=0.0
            offset_00=0.0
            offset_11=0.0
        simPhi_01s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_01s)):
            gamma=getGamma(K_01s[kd],sqrtK00K11) #current
            simPhi_01_temp=1.0/(2.0*jnp.pi)*sqrtK00K11*convert(gamma)
            simPhi_01s.append(simPhi_01_temp)
        simPhi_01=jnp.array(simPhi_01s[0])*1.0
        K_01s=[]
        for kd in range(zh+1):
            S=jnp.zeros((s0,s1))
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_01s)-1))
                S+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_01s[id]-offset_01)
            K_01_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_01+2*((sigma**2)/(zh*zw))*S + sigmab**2#next
            #K_01s.append(jnp.array(np.copy(K_01_temp))) #next
            K_01s.append(K_01_temp)

        np.save(path_name+'_L{}'.format(i),K_01s[0])
        print('{}th layer kernel saved'.format(i))
        load_names.append(path_name+'_L{}'.format(i)+'.npy')
        
        ###for K00s and K11s
        simPhi_00s=[]
        simPhi_11s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_00s)):
            gamma=getGamma(K_00s[kd],K_00s[0])
            simPhi_00_temp=1.0/(2.0*jnp.pi)*K_00s[0]*convert(gamma) #current

            gamma=getGamma(K_11s[kd],K_11s[0])
            simPhi_11_temp=1.0/(2.0*jnp.pi)*K_11s[0]*convert(gamma) #current
            #print(simPhi_11_temp-jnp.diag(simPhi_01s[kd]))
            simPhi_00s.append(simPhi_00_temp)
            simPhi_11s.append(simPhi_11_temp)
        simPhi_00=jnp.array(simPhi_00s[0])*1.0
        simPhi_11=jnp.array(simPhi_11s[0])*1.0

        K_00s=[]
        K_11s=[]
        for kd in range(zh+1):
            S00=jnp.zeros(s0)
            S11=jnp.zeros(s1)
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_00s)-1))
                S00+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_00s[id]-offset_00)
                S11+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_11s[id]-offset_11)
            K_00_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_00+2*((sigma**2)/(zh*zw))*S00 + sigmab**2#next
            K_11_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_11+2*((sigma**2)/(zh*zw))*S11 + sigmab**2#next
            #print(K_11_temp-jnp.diag(K_01s[kd]))
            #K_00s.append(jnp.array(np.copy(K_00_temp)))
            #K_11s.append(jnp.array(np.copy(K_11_temp)))
            K_00s.append(K_00_temp)
            K_11s.append(K_11_temp)
    print('done')
    return load_names
    #return K_01s[0],arcK(K_01s[0],jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]))

#If L=2, K_01sL will have two elements. The first hidden layer preactivation cov. and one after that.
#In the main code, use Ls=[1,2,3,...] (start from 1)
#When performing a deep Hebbian learning, ignore the last element of K_01sL.
#The first element of K_01sL and S_01sL are from the first hidden layer, and so on.
#S_01sL does not include the last layer post-activation similarity. So K_01sL is 1 element longer than S_01sL.
def getGPprior_for_HAL(x0,x1,mw0=0,mh0=0,mw=4,mh=10,sigma=1.0,taus=2.0,L=2,ntrapz=1000,chunk_size=100000,layer_zero_mean=False,input_unit_ball=True):
    if layer_zero_mean:
        x0=x0-jnp.mean(x0,axis=1)[:,None]
        x1=x1-jnp.mean(x0,axis=1)[:,None]
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))
    #ntrapz=200
    #chunk_size=200000
    if np.shape(taus)==():
        taus=np.ones(L-1)*taus
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]
    #initialize
    K_01s=[]
    zw=mw+1
    zh=mh+1
    for kd in range(mh0+1+1):
        K_01s.append(getKab1(x0.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd))
    K_00s=[]
    for kd in range(mh0+1+1):
        K_00s.append(jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_11s=[]
    for kd in range(mh0+1+1):
        K_11s.append(jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
    K_01sL=[]
    P_01sL=[]
    K_01sL.append(jnp.array(np.copy(K_01s[0])))
    S_01sL=[]
    for i in range(L-1):
        tau=taus[i]
        sqrtK00K11=jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]) #current
        simPhi_01s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_01s)):
            theta=arcK(K_01s[kd],sqrtK00K11) #current
            I=I2_chunk(theta.flatten(),jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            I=jnp.reshape(I,jnp.shape(theta))
            simPhi_01_temp=1.0/(2.0*jnp.pi)*sqrtK00K11*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current
            simPhi_01s.append(simPhi_01_temp)
        simPhi_01=jnp.array(simPhi_01s[0])*1.0
        S_01sL.append(simPhi_01)
        K_01s=[]
        for kd in range(zh+1):
            S=jnp.zeros((s0,s1))
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_01s)-1))
                S+=f(0,0,kd,gd+1,mw=mw,mh=mh)*simPhi_01s[id]
            K_01_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_01+2*((sigma**2)/(zh*zw))*S #next
            K_01s.append(jnp.array(np.copy(K_01_temp))) #next
        K_01sL.append(jnp.array(np.copy(K_01s[0])))

        ###for K00s and K11s
        simPhi_00s=[]
        simPhi_11s=[]
        #for kd in range(zh+1):
        for kd in range(len(K_00s)):
            theta=arcK(K_00s[kd],K_00s[0])
            I=I2_chunk(theta,jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            simPhi_00_temp=1.0/(2.0*jnp.pi)*K_00s[0]*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current

            theta=arcK(K_11s[kd],K_11s[0])
            I=I2_chunk(theta,jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
            simPhi_11_temp=1.0/(2.0*jnp.pi)*K_11s[0]*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current
            #print(simPhi_11_temp-jnp.diag(simPhi_01s[kd]))
            simPhi_00s.append(simPhi_00_temp)
            simPhi_11s.append(simPhi_11_temp)
        simPhi_00=jnp.array(simPhi_00s[0])*1.0
        simPhi_11=jnp.array(simPhi_11s[0])*1.0

        K_00s=[]
        K_11s=[]
        for kd in range(zh+1):
            S00=jnp.zeros(s0)
            S11=jnp.zeros(s1)
            for gd in range(mw):
                id=np.min((gd+1,len(simPhi_00s)-1))
                S00+=f(0,0,kd,gd+1,mw=mw,mh=mh)*simPhi_00s[id]
                S11+=f(0,0,kd,gd+1,mw=mw,mh=mh)*simPhi_11s[id]
            K_00_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_00+2*((sigma**2)/(zh*zw))*S00 #next
            K_11_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_11+2*((sigma**2)/(zh*zw))*S11 #next
            #print(K_11_temp-jnp.diag(K_01s[kd]))
            K_00s.append(jnp.array(np.copy(K_00_temp)))
            K_11s.append(jnp.array(np.copy(K_11_temp)))
    return K_01sL[:-1],S_01sL #actually here we are popping out the last element of K_01sL so that its length matches that of S_01sL

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def varsize():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        
        
        
### jax autograd-able

def I2_gradable(thetar,taur,n=100):
    theta=thetar[None,:,None]
    xi=(jnp.pi-thetar)/2.0
    tau=taur[:,None,None]
    g=1.0
    phi=jnp.linspace(1e-4,xi,n).T[None,:,:]
    L_term1=2*jnp.sin(phi+theta)*jnp.sin(phi)*jnp.exp(-0.5*jnp.square(g*tau/jnp.sin(phi)))
    L_term2=g*tau*(jnp.sin(phi+theta)+jnp.sin(phi))*jnp.sqrt(jnp.pi/2)*jsc.special.erf(jnp.sqrt(0.5)*g*tau/jnp.sin(phi))
    inte=jnp.trapz(L_term1+L_term2,x=phi,axis=-1)
    return inte

    
def I2_chunk_gradable(theta,tau,n=1000,chunk_size=10000):
    N=jnp.shape(theta)[0]
    n_chunks=int(jnp.ceil(N/chunk_size))
    
    remain=jnp.remainder(N,chunk_size)

    pad_size=(chunk_size-remain) * jnp.invert(remain==0)
    thetas=jnp.reshape(jnp.concatenate([theta,jnp.zeros(pad_size)]),(n_chunks,chunk_size))
        
    inte=[]
    for i,theta_c in enumerate(thetas):
        ival=I2_gradable(theta_c,tau,n=n)
        inte.append(jnp.squeeze(ival))
    inte=jnp.concatenate(inte)[:N]
    return inte

def getSCK_gradable(x0,x1,mw0=0,mh0=0,mw=0,mh=0,sigma=1.0,L=2,ntrapz=1000,chunk_size=100000):
    d=jnp.shape(x1)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]

    offset_01=0.0
    offset_00=0.0
    offset_11=0.0
    
    def getKm(tau):
        #initialize
        K_01s=[]
        zw=mw+1
        zh=mh+1
        for kd in range(mh0+1+1):
            K_01s.append(getKab1(x0.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd))
        K_00s=[]
        for kd in range(mh0+1+1):
            K_00s.append(jnp.diag(getKab1(x0.T,x0.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
        K_11s=[]
        for kd in range(mh0+1+1):
            K_11s.append(jnp.diag(getKab1(x1.T,x1.T,sigma=sigma,mw=mw0,mh=mh0,id=0,kd=kd)))
        #P_01sL=[]
        SK_L=[]
        for i in range(L-1):
            sqrtK00K11=jnp.sqrt(K_00s[0][:,None]*K_11s[0][None,:]) #current
            simPhi_01s=[]
            #for kd in range(zh+1):
            for kd in range(len(K_01s)):
                theta=arcK(K_01s[kd],sqrtK00K11) #current
                I=I2_chunk_gradable(theta.flatten(),jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
                I=jnp.reshape(I,jnp.shape(theta))
                simPhi_01_temp=1.0/(2.0*jnp.pi)*sqrtK00K11*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current
                simPhi_01s.append(simPhi_01_temp)
            simPhi_01=jnp.array(simPhi_01s[0])*1.0

            K_01s=[]
            for kd in range(zh+1):
                S=jnp.zeros((s0,s1))
                for gd in range(mw):
                    id=np.min((gd+1,len(simPhi_01s)-1))
                    S+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_01s[id]-offset_01)
                K_01_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_01+2*((sigma**2)/(zh*zw))*S #next
                K_01s.append(K_01_temp) #next
            SK_L.append(K_01s[0])

            ###for K00s and K11s
            simPhi_00s=[]
            simPhi_11s=[]
            #for kd in range(zh+1):
            for kd in range(len(K_00s)):
                theta=arcK(K_00s[kd],K_00s[0])
                I=I2_chunk_gradable(theta,jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
                simPhi_00_temp=1.0/(2.0*jnp.pi)*K_00s[0]*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current

                theta=arcK(K_11s[kd],K_11s[0])
                I=I2_chunk_gradable(theta,jnp.array([tau]),n=ntrapz,chunk_size=chunk_size)
                simPhi_11_temp=1.0/(2.0*jnp.pi)*K_11s[0]*(2.0*I-tau*jnp.sqrt(2.0*jnp.pi)*(1.0+jnp.cos(theta))) #current
                #print(simPhi_11_temp-jnp.diag(simPhi_01s[kd]))
                simPhi_00s.append(simPhi_00_temp)
                simPhi_11s.append(simPhi_11_temp)
            simPhi_00=jnp.array(simPhi_00s[0])*1.0
            simPhi_11=jnp.array(simPhi_11s[0])*1.0

            K_00s=[]
            K_11s=[]
            for kd in range(zh+1):
                S00=jnp.zeros(s0)
                S11=jnp.zeros(s1)
                for gd in range(mw):
                    id=np.min((gd+1,len(simPhi_00s)-1))
                    S00+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_00s[id]-offset_00)
                    S11+=f(0,0,kd,gd+1,mw=mw,mh=mh)*(simPhi_11s[id]-offset_11)
                K_00_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_00+2*((sigma**2)/(zh*zw))*S00 #next
                K_11_temp=((sigma**2)/(zh*zw))*f(0,0,kd,0,mw=mw,mh=mh)*simPhi_11+2*((sigma**2)/(zh*zw))*S11 #next
                #print(K_11_temp-jnp.diag(K_01s[kd]))
                K_00s.append(K_00_temp)
                K_11s.append(K_11_temp)
        return SK_L
    
    return getKm


########## For Neurips 2023 rebuttal

#### Cho and Saul 's Heaviside sparsity
def IH2(rr,zetar,n=100):
    r=rr[:,None]
    #zetar[:,None,None]
    phi=jnp.linspace(1e-20,zetar,n).T
    L_term=jnp.exp(-1/(2*jnp.square(r)*jnp.square(jnp.sin(phi))))
    inte=(1/jnp.pi)*jnp.trapz(L_term,x=phi,axis=-1)
    return inte

def IH2_chunk(theta,zeta,n=1000,chunk_size=10000,silence=False):
    N=jnp.shape(theta)[0]
    n_chunks=int(jnp.ceil(N/chunk_size))
    if N != chunk_size:
        remain=N%chunk_size
        if remain==0:
            pad_size=0
        else:
            pad_size=chunk_size-N%chunk_size
        thetas=jnp.reshape(jnp.concatenate([theta,jnp.zeros(pad_size)]),(n_chunks,chunk_size))
        zetas=jnp.reshape(jnp.concatenate([zeta,jnp.zeros(pad_size)]),(n_chunks,chunk_size))
    else:
        #thetas=jnp.array([theta])
        thetas=theta*1.0
        zetas=zeta*1.0
        
    inte=[]
    for i,theta_c in enumerate(thetas):
        ival=IH2(theta_c,zetas[i],n=n)
        inte.append(jnp.squeeze(ival))
        if not silence:
            print('\rProgress={:.2f}%, #chunks:{}'.format(100*i/n_chunks,n_chunks),end='')
    inte=jnp.concatenate(inte)[:N]
    #print('')
    return inte

def getHSK(x0,x1,bs=1,L=2,ntrapz=1000,chunk_size=100000,input_unit_ball=True,f=0.2,fixSparse=False):
    if fixSparse:
        tau=jnp.sqrt(2.0)*jsc.special.erfinv(1.0-2.0*f)
    if input_unit_ball:
        x0=x0/jnp.sqrt(jnp.sum(jnp.square(x0),axis=1))[:,None]
        x1=x1/jnp.sqrt(jnp.sum(jnp.square(x1),axis=1))[:,None]     

    d=jnp.shape(x0)[1]
    s0=jnp.shape(x0)[0]
    s1=jnp.shape(x1)[0]

    if np.isscalar(bs):
        bs=bs*np.ones(L-1)
    
    K_00=jnp.mean(jnp.square(x0),axis=1)
    K_11=jnp.mean(jnp.square(x1),axis=1)
    K_01=1/d*jnp.matmul(x0,x1.T)
    SK_L=[]
    for i in range(L-1):
        if fixSparse:
            b0=tau*jnp.sqrt(K_00)
            b1=tau*jnp.sqrt(K_11)
        else:
            b0=bs[i]
            b1=bs[i]

        r0=1/b0*jnp.sqrt(K_00)
        r0=jnp.tile(r0,(s1,1)).T #vertical v0's
        zeta0=jnp.arccos((K_00[:,None]-K_01)/(jnp.sqrt(K_00[:,None]*(K_00[:,None]+K_11[None,:]-2*K_01))))
        zeta0=jnp.where(jnp.isnan(zeta0),jnp.pi/2,zeta0)
        I0=IH2_chunk(r0.flatten(),zeta0.flatten(),n=ntrapz,chunk_size=chunk_size)
        I0=jnp.reshape(I0,jnp.shape(zeta0))

        r1=1/b1*jnp.sqrt(K_11)
        r1=jnp.tile(r1,(s0,1)) #horizontal v1's
        zeta1=jnp.arccos((K_11[None,:]-K_01)/(jnp.sqrt(K_11[None,:]*(K_11[None,:]+K_00[:,None]-2*K_01))))
        zeta1=jnp.where(jnp.isnan(zeta1),jnp.pi/2,zeta1)
        I1=IH2_chunk(r1.flatten(),zeta1.flatten(),n=ntrapz,chunk_size=chunk_size)
        I1=jnp.reshape(I1,jnp.shape(zeta1))
        K_01=jnp.squeeze(I0+I1)
        
        r00=1/b0*jnp.sqrt(K_00)
        zeta00=jnp.ones(s0)*jnp.pi/2
        I00=IH2_chunk(r00,zeta00,n=ntrapz,chunk_size=chunk_size)
        K_00=2*I00
        


        r11=1/b1*jnp.sqrt(K_11)
        zeta11=jnp.ones(s1)*jnp.pi/2
        I11=IH2_chunk(r11,zeta11,n=ntrapz,chunk_size=chunk_size)
        K_11=2*I11
        
        SK_L.append(K_01)
    return SK_L
