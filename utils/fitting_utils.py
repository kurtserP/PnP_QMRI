import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit, minimize

def fitting_estimation_plot(b,y,S0, MD,p, indi=120,indj=120,idslice=10):
    # p0 = find_initial_guess_qMRI_S0MD(np.asarray(b), y[indi, indj, idslice, :])
    plt.scatter(np.asarray(b), y[indi, indj, idslice, :])
    # plt.plot(np.asarray(b), p0[0] * math.e ** (-np.asarray(b) * p0[1]), color='steelblue', linestyle='--', linewidth=2)
    # p, pcov = curve_fit(qMRI_S0MD, b, y[indi, indj, idslice, :], p0)
    # res = minimize(ADMM_cost, p0, args=(np.asarray(b), y[indi, indj, idslice, :], 0, 1))
    # p = res.x
    plt.plot(np.asarray(b), p[indi, indj, idslice, 0] * math.e ** (-np.asarray(b) * p[indi, indj, idslice, 1]), color='red', linestyle='--', linewidth=2)
    plt.plot(np.asarray(b), S0[indi, indj, idslice] * math.e ** (-np.asarray(b) * (MD[indi, indj, idslice])), color='green',
             linestyle='--', linewidth=2)
    plt.legend(["Datapoints", "Fitting", "Ground truth"])
    plt.show()



def fit_parameters(b, y,masks,MD_=[],mu=0):
    if np.size(MD_) == 0:
        MD_= np.zeros((np.shape(y)[0], np.shape(y)[1], np.shape(y)[2]))
    p = np.empty((np.shape(y)[0], np.shape(y)[1], np.shape(y)[2], 2))
    p[:] = np.nan
    for slice_no in range(np.shape(y)[2]):
        print("Fitting slice no: " + str(slice_no))
        for i in range(np.shape(y)[0]):
            for j in range(np.shape(y)[1]):
                if np.any(np.isnan(y[i, j, slice_no, :])):
                    pass
                elif masks[i, j, slice_no] ==0:
                    p[i, j, slice_no, :] = [0,0]
                else:
                    p0 = find_initial_guess_qMRI_S0MD(np.asarray(b), y[i, j, slice_no, :])
                    try:
                            if(np.any(np.isnan(p0))):
                                # p[i, j, slice_no, :], pcov = curve_fit(qMRI_S0MD, b, y[i, j, slice_no, :])
                                pass
                            else:
                                # p[i, j, slice_no, :], pcov = curve_fit(qMRI_S0MD, b, y[i, j, slice_no, :], p0)
                                res= minimize(ADMM_cost, p0, args=(np.asarray(b), y[i, j, slice_no, :], MD_[i, j, slice_no],mu))
                                p[i, j, slice_no, :] = res.x
                    except RuntimeError:
                        pass

    return p


def find_initial_guess_qMRI_S0MD(x, y):
    # for y = b*exp(cx) - find b,c
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    n = len(x)
    Sk = np.zeros(n)
    for k in range(n-1):
        Sk[k+1] = Sk[k]+(y[k+1]+y[k])*(x[k+1]-x[k])

    H = [[np.sum(Sk**2), np.sum(Sk)], [np.sum(Sk), n]]
    if np.linalg.det(H):
        c = float(np.matmul(np.linalg.inv([[np.sum(Sk**2), np.sum(Sk)], [np.sum(Sk), n]]), [[np.sum(Sk*y)], [np.sum(y)]])[0])
        b = np.sum(y)/np.sum(np.exp(c*x))
        p0 = [b, -c]
    else:
        p0 = np.nan

    return p0

def qMRI_S0MD(bi, S0, MD):
    y = S0*np.exp(-bi*MD)
    return y

def ADMM_cost(p,b,yij,MD_,mu):
    S0, MD = p
    # model = np.zeros(np.shape(b))
    # for idx, bi in enumerate(b):
    model= qMRI_S0MD(b, S0, MD)
    return np.mean((model - yij) ** 2)+mu*np.abs(MD-MD_)+0.0001*S0 ** 2 + 1*np.abs(MD)

def reconstruct_y(b, S0_, MD_, shape_y):
    y_ = np.zeros(shape_y)
    for slice_no in range(shape_y[2]):
        for idx, bi in enumerate(b):
            for i in range(shape_y[0]):
                for j in range(shape_y[1]):
                    y_[i, j, slice_no, idx] = qMRI_S0MD(bi, S0_[i, j, slice_no], MD_[i, j, slice_no])

    return y_