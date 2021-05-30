""""
EM learning for states space model
"""
import numpy as np 
from numpy.linalg import pinv
from ssm_kalman import * 

def E_step(X, y_init, Q_init, A, Q, C, R):
    ## compute <y_tˆ{T}.
    ## y_hat = <y_tˆ{T}> , t = 1 ... 1000
    y_hat, V_hat, V_joint, loglikelihood \
        = run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='smooth')
    ## retrive T
    _, T = y_hat.shape 
    ## compute <y_t, y_tˆ{T}>
    ## P_t = <y_t transpose(y_t)> = V_tˆ{T} + <y_t> transposre(<y_t>)
    P = V_hat + y_hat.T[:, :, None] * y_hat.T[:, None]
    ## compute <y_t y_{t-1}ˆ{T}>
    ## P_joint = <y_t transposre(y_{t-1})> = V_{t,t-1}ˆ{T} + <y_t> transposre(<y_{t-1}>)
    P_joint = V_joint[0:T-1] + y_hat.T[1:T][:, :, None] * y_hat.T[:T-1][:, None]

    return y_hat, P, P_joint, loglikelihood

def M_step(X, y_hat, P, P_joint):

    _, T = y_hat.shape

    """"
    xtyt: \sum_{t=1}ˆ{T} x_t transpose(<y_t>)        [5,4] np array
    xyxt: \sum_{t=1}ˆ{T} x_t transpose(x_t)          [5,5] np array
    ytyt: \sum_{t=1}ˆ{T} <y_t transpose(y_t)>        [4,4] np array
    ytyt_2: \sum_{t=1}ˆ{T-1} <y_t transpose(y_t)     [4,4] np array
    ytyt_3: \sum_{t=2}ˆ{T} <y_t transpose(y_t)>      [4,4] np array
    P_joint: <y_t transpose(y_{t-1})>
    """
    xtyt = np.sum((X[:, :, None] * y_hat.T[:, None]), axis=0)
    xtxt = np.sum(X[:, :, None] * X[:, None], axis=0)
    ytyt = np.sum(P, axis=0)
    ytyt_2 = np.sum(P[0:T-1], axis=0)
    ytyt_3 = np.sum(P[1:T], axis=0)

    C_new = xtyt.dot(pinv(ytyt))
    R_new = (1/T) * (xtxt - xtyt.dot(C_new.T))
    A_new = np.sum(P_joint, axis=0).dot(pinv(ytyt_2))
    Q_new = (1/(T-1)) * (ytyt_3 - (np.sum(P_joint, axis=0)).dot(A_new))

    return C_new, R_new, A_new, Q_new

def EM_LGSSM(X, A0, Q0, C0, R0, max_iter):

    A = A0
    Q = Q0
    C = C0
    R = R0

    loglikeStore = [] 

    mean_init = np.zeros(A.shape[0])
    cov_init = np.identity(A.shape[0])
    ## random k-dim multivariate gaussian
    y_init = np.random.multivariate_normal(mean_init, cov_init)
    Q_init = cov_init

    for i in range(max_iter):
        y_hat, P_hat, P_joint_hat, loglike = \
            E_step(X.T, y_init, Q_init, A, Q, C, R)
        loglikeStore.append(np.sum(loglike))
        C, R, A, Q = M_step(X, y_hat, P_hat, P_joint_hat)
    
    return loglikeStore[1:max_iter]
