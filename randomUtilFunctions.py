"""
Miscellaneous functions, of use in multiple contexts
"""
import numpy as np
from scipy import signal

def attributesFromDict(d):
    """
    Set attributes from input list automatically
    """
    self = d.pop('self')
    for n, v in d.items():
        setattr(self, n, v)


def logLogExpFit(dataVec,firstPointPercent=0.2,secondPointPercent=0.6):
    """
    Fit a log-linear model to data, of the form log(dataVec) = a+b*log(idx)
    :param dataVec:
    :param firstPointPercent:
    :param secondPointPercent:
    :return:
    """
    firstIdx = round(firstPointPercent*len(dataVec))
    lastIdx = round(secondPointPercent*len(dataVec))
    logData = np.log(abs(dataVec[firstIdx:lastIdx]))
    logData = np.reshape(logData,(len(logData),1))

    projVecs = np.zeros((len(logData),2))
    projVecs[:,0] = np.ones(len(logData))
    projVecs[:,1] = np.log(range(firstIdx,lastIdx))

    projComps = np.linalg.lstsq(projVecs,logData)
    logLogExp = projComps[0][1]
    return logLogExp


def logDeriv(xVals,funVal,alpha=0):
    """
    Logarithmic derivative of f(x) (numerical). Optional gaussian convolution to smooth.

    :param xVal:
    :param funVal:
    :param alpha: optional parameter for Gaussian smoothing. Standard deviation of gaussian used.
    :return outXvals: output x values; midpoints of current x values
    :return derivVals: Length is len(funVal)-1
    """
    derivVals = np.zeros(len(xVals))
    outXVals = np.log((xVals[0:-1]+xVals[1:])/2)
    if alpha==0:
        # no convolution
        derivVals = (np.log(funVal[1:])-np.log(funVal[0:-1]))/(np.log(xVals[1:])-np.log(xVals[0:-1]))
    else:
        # with smoothing
        weightVec = signal.gaussian(len(funVal),alpha)/(alpha*np.sqrt(2*np.pi))
        adjFunVal = signal.fftconvolve(funVal,weightVec,mode="same")
        derivVals = (np.log(adjFunVal[1:])-np.log(adjFunVal[0:-1]))/(np.log(xVals[1:])-np.log(xVals[0:-1]))
    return derivVals,outXVals

def monotoneRootBisect(rootFun,x0=0,xTol=1e-2,yTol=1e-4,initStepSize=1,sense="increasing",maxIters=100):
    """

    :param rootFun:
    :param x0:
    :param xTol:
    :param yTol:
    :param initStepSize:
    :param sense: direction
    :return rootVal: root value
    """
    if sense=="increasing":
        dirSign = -1
    else:
        dirSign = 1
    curStepSize = initStepSize+xTol
    curY = yTol+1
    numIters = 0
    curX = x0
    curSign = 1

    while curStepSize>xTol and abs(curY)>yTol and numIters<maxIters:
        curX = curX+curSign*curStepSize
        curY = rootFun(curX)
        nextSign = np.sign(curY)*dirSign
    return curX

def branchingPDF(x,a=1,b=1):
    """
    Approximate PDF for branching process. Mean is a, variance 2ab
    :param x: value to evaluate at
    :param a: mean
    :param b: 1/2 var/mean
    :return pdfVal: pdf value at x
    """
    if x>=1:
        pdfVal = np.sqrt(np.sqrt(a)/(4*np.pi*b*(np.power(abs(x),1.5))))*np.exp(-(np.power((np.sqrt(x)-np.sqrt(a)),2))/b)
    else:
        pdfVal = np.sqrt(np.sqrt(a)/(4*np.pi*b))*np.exp(-(np.power((np.sqrt(x)-np.sqrt(a)),2))/b)
    return pdfVal