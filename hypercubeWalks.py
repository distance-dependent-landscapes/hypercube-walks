"""
Python module for simulation of random walk on a random fitness landscape.
The landscape is drawn as you walk; landscape is assumed to be high enough dimensional that
structure is locally tree-like. This means that single reversions allowed, but multiple paths
to same genotype are not.
"""

import numpy as np
from scipy import integrate as integrate
from scipy.stats import norm as norm
from scipy.optimize import minimize as minimize
from scipy.optimize import brentq as brentq
from scipy.optimize import fsolve as fsolve
from randomUtilFunctions import monotoneRootBisect
from copy import deepcopy,copy

"""
Data structures. Edge class and children. Store trees as vectors of edge objects.
"""


class edge:
    """
    Base edge class. Has following properties:
    hasValue - indicator of whether or not value assigned
    treeAddress - vector of integers, addresses location in tree
    edgeValue - the value of the edge (once drawn)
    treeIdx - index in the tree; used for lookups of correlation matrix
    assignmentTime - real time at which edge assignment was made
    """
    def __init__(self,treeAddress=None,edgeVal=None,treeIdx=None,assignmentTime=None):
        self.treeAddress = treeAddress
        self.edgeVal = edgeVal
        self.hasValue = False
        self.treeIdx = treeIdx
        self.assignmentTime = assignmentTime
        if edgeVal!= None:
            self.hasValue = True

    def setVal(self,edgeVal):
        self.edgeVal = edgeVal
        self.hasValue = True


class meanEdge(edge):
    """
    Mean edge. Inherits all edge properties. Also has an associated variance.
    """
    edgeType = 'mean'
    def __init__(self,edgeVar=None,**kwargs):
        edge.__init__(self,**kwargs)
        self.edgeVar = edgeVar

    def genMeanVal(self,regTree,meanTree,corrMat):
        """
        Generate random mean edge, given past

        :param regTree: tree values for regular edges
        :param meanTree: tree values for mean edges
        :param corrMat: correlation matrix, current mean edge first, then reg, then mean
        :return meanVal: value of mean edge
        """

        # Generate variance and response functions

        # variance
        corVar = corrMat[0,0]
        if np.size(corrMat)>1:
            # conditional correlation matrix
            subCorrMat = corrMat[1:,1:]
            # conditional vector
            subCorrVec = corrMat[0,1:]

            # response function
            responseFun = np.linalg.solve(subCorrMat,subCorrVec)

            # response values
            numRegEdges = len(regTree)
            numMeanEdges = len(meanTree)
            edgeVals = np.zeros(numRegEdges+numMeanEdges)
            for i in range(0,numRegEdges):
                edgeVals[i] = regTree[i].edgeVal
            for i in range(numRegEdges,numRegEdges+numMeanEdges):
                edgeVals[i] = meanTree[i-numRegEdges].edgeVal

            # calculate variance of mean
            meanVar = corVar-np.dot(subCorrVec,responseFun)

            # calculate mean of mean
            meanMean = np.dot(edgeVals,responseFun)

            # Draw mean value
            meanVal = np.random.normal(loc=meanMean,scale=np.sqrt(meanVar))
        else:
            meanVal = np.random.normal(scale=np.sqrt(corrMat[0,0]))

        self.setVal(meanVal)
        return meanVal

    def setValWithResponse(self,regTree,meanTree,responseFuns):
        """
        Use response function to conditionally draw mean of DFE
        :param regTree:
        :param meanTree:
        :param responseFuns:
        :return: mean value of DFE
        """
        meanMean = 0
        # regular vals contribution
        for i,idxVal in enumerate(responseFuns[0][0]):
            meanMean = meanMean+responseFuns[0][1][i]*regTree[int(idxVal)].edgeVal
        # mean vals contribution
        for i,idxVal in enumerate(responseFuns[1][0]):
            meanMean = meanMean+responseFuns[1][1][i]*meanTree[int(idxVal)].edgeVal
        meanVal = np.random.normal(loc=meanMean,scale=np.sqrt(self.edgeVar))
        self.setVal(meanVal)
        return meanVal


class regEdge(edge):
    """
    reg edge. Inherits all edge properties, and has additional pointer to its base mean edge.
    """
    edgeType = 'reg'
    def __init__(self,baseMean=None,edgeVar=None,**kwargs):
        edge.__init__(self,**kwargs)
        self.baseMean = baseMean
        self.edgeVar = edgeVar

    def setMeanEdge(self,baseMean):
        self.baseMean = baseMean


"""
Edge methods
"""

def edgeCorr(edge1,edge2,corrFun):
    """
    edgeCorr - correlation between two edges
    :param edge1: first edge
    :param edge2: second edge
    :param corrFun: correlation function
    :return corrVal: correlation between edges
    """

    baseDist = getDist(edge1,edge2)
    corrVal = corrFun(baseDist)

    return corrVal


def getDist(edge1,edge2):
    """
    Get oriented distance between two edges

    :param edge1: First edge
    :param edge2: Second edge
    :return baseDist: Distance between edges. Measured from tip to tip; oriented
    """

    # Find most recent common ancestor
    treeAddress1 = edge1.treeAddress
    treeAddress2 = edge2.treeAddress

    minLen = min(len(treeAddress1),len(treeAddress2))
    ancestorIdx = 0
    ancestorFound = False


    while ancestorFound == False:
        if ancestorIdx<minLen:
            if treeAddress1[ancestorIdx]==treeAddress2[ancestorIdx]:
                ancestorIdx = ancestorIdx+1
            else:
                ancestorFound = True
        else:
            ancestorFound = True

    # distance to nearest common ancestor
    distToAncestor = len(treeAddress1)+len(treeAddress2)-2*ancestorIdx

    # Break up into cases by edge types

    if edge1.edgeType=='reg' and edge2.edgeType=='reg':
        if len(treeAddress1)==ancestorIdx or len(treeAddress2)==ancestorIdx:
            # direct descent
            baseDist = distToAncestor
        else:
            baseDist = -distToAncestor+1
    elif edge1.edgeType=='mean' and edge2.edgeType=='mean':
        baseDist = -distToAncestor-1
    else:
        if len(treeAddress1)==ancestorIdx or len(treeAddress2)==ancestorIdx:
            # direct descent
            if edge1.edgeType=='mean' and len(treeAddress1) >= len(treeAddress2):
                baseDist = distToAncestor+1
            elif edge1.edgeType=='mean' and len(treeAddress1) < len(treeAddress2):
                baseDist = -distToAncestor
            elif edge2.edgeType=='mean' and len(treeAddress2) >= len(treeAddress1):
                baseDist = distToAncestor+1
            elif edge2.edgeType=='mean' and len(treeAddress2) < len(treeAddress1):
                baseDist = -distToAncestor
        else:
            baseDist = -distToAncestor



    return baseDist


def genEdgeVal(meanVal,varVal,weightFun):
    """
    Generate next step, given mean, var, and weighting function

    :param meanVal: mean of steps
    :param varVal: variance of steps
    :param weightFun: weighting function for steps. Can be string for special method.
    :return edgeVal: value of edge
    """
    # Special methods

    # uphill walk
    if weightFun=="uphillWalk":
        # Calculate P(edgeVal>0)
        zScale = norm.cdf(meanVal/np.sqrt(varVal))
        # Draw from conditional distribution
        zVal = zScale*np.random.rand()
        edgeVal = -norm.ppf(zVal)*np.sqrt(varVal)+meanVal
    elif weightFun=="constant":
        edgeVal = np.sqrt(varVal)*np.random.rand()+meanVal
    else:
        def unNormPDF(x):
            #return norm.pdf((x-meanVal)/np.sqrt(varVal))*weightFun(x)
            return np.exp(-np.power(x-meanVal,2)/(2*varVal))*weightFun(x)

        normFactor,normError = integrate.quad(unNormPDF,-np.Inf,np.Inf)

        def choiceCDF(x):
            integralVal,integralError = integrate.quad(unNormPDF,-np.Inf,x)
            integralVal = integralVal/normFactor
            return integralVal

        # inverse CDF method
        Z = np.random.rand()
        def funToInvert(x):
            return choiceCDF(x)-Z
        edgeVal = monotoneRootBisect(funToInvert)
    return edgeVal


"""
Tree methods
"""

def cleanTree(edgeTree):
    """
    Erase all tree value references
    :param edgeTree:
    :return:
    """
    for i in range(0,len(edgeTree)):
        edgeTree[i].hasValue = False
    return edgeTree


def copyTrees(regTree,meanTree):
    """
    Copy trees. Deep copy, with proper referencing
    :param regTree:
    :param meanTree:
    :return:
    """
    meanTreeCopy = [None]*len(meanTree)
    regTreeCopy = deepcopy(regTree)
    for i,regEdge in enumerate(regTreeCopy):
        meanTreeIdx = regEdge.baseMean.treeIdx
        meanTreeCopy[meanTreeIdx] = regEdge.baseMean
    return regTreeCopy,meanTreeCopy


def copySubTree(regTree,regTreeIdx):
    """
    Copy subtrees. Deep copy.
    :param regTree:
    :param meanTree:
    :return:
    """

    regTreeCopy = [None]*len(regTreeIdx)
    meanTreeCopy = [None]*len(regTreeIdx)
    meanIdx = 0
    for i,idx in enumerate(regTreeIdx):
        regTreeCopy[i] = deepcopy(regTree[idx])
        regTreeCopy[i].treeIdx = i
        if regTreeCopy[i].baseMean.treeIdx>=0:
            regTreeCopy[i].baseMean.treeIdx = -1-meanIdx
            meanTreeCopy[meanIdx] = regTreeCopy[i].baseMean
            meanIdx = meanIdx+1


    meanTreeCopy = meanTreeCopy[0:meanIdx]

    for i,meanEdge in enumerate(meanTreeCopy):
        meanEdge.treeIdx = i

    return regTreeCopy,meanTreeCopy


def pathVals(edgeTree):
    """
    Extract vector of values from trees
    :param edgeTree:
    :return treeVals:
    """
    treeVals = np.zeros(len(edgeTree))

    for i,val in enumerate(treeVals):
        treeVals[i] = edgeTree[i].edgeVal

    return treeVals


"""
Correlation matrix methods
"""

def genCorrMat(regTree,meanTree,corrFun,straightPath=False,maxDist=None):
    """
    genCorrMat - generate correlation matrix given known edges
    :param regTree: all regular edges
    :param meanTree: all mean edges
    :param corrFun: correlation function
    :param straightPath: whether or not to use quick straight path algorithm
    :return: corrMat: correlation matrix. Ordering is: regTree, meanTree
    """
    if straightPath==False:
        regLen = len(regTree)
        meanLen = len(meanTree)
        corrMatSize = regLen+meanLen
        corrMat = np.zeros((corrMatSize,corrMatSize))
        if maxDist==None:
            for i in range(0,corrMatSize):
                # choosing edge types. reg edges first, then mean edges
                for j in range(0,corrMatSize):
                    if i<regLen and j<regLen:
                        edge1 = regTree[i]
                        edge2 = regTree[j]
                    elif i>=regLen and j<regLen:
                        edge1 = meanTree[i-regLen]
                        edge2 = regTree[j]
                    elif i>=regLen and j>=regLen:
                        edge1 = meanTree[i-regLen]
                        edge2 = meanTree[j-regLen]
                    elif i<regLen and j>=regLen:
                        edge1 = regTree[i]
                        edge2 = meanTree[j-regLen]
                    # Calculate distance, then use (signed) correlation function to calculate
                    baseDist = getDist(edge1,edge2)
                    corrMat[i,j] = corrFun(baseDist)
        else:
            corrFunVec = np.zeros(2*maxDist+1)
            for i in range(-maxDist,maxDist+1):
                corrFunVec[i+maxDist] = corrFun(i)

            for i in range(0,corrMatSize):
                # choosing edge types. reg edges first, then mean edges
                for j in range(0,corrMatSize):
                    if i<regLen and j<regLen:
                        edge1 = regTree[i]
                        edge2 = regTree[j]
                    elif i>=regLen and j<regLen:
                        edge1 = meanTree[i-regLen]
                        edge2 = regTree[j]
                    elif i>=regLen and j>=regLen:
                        edge1 = meanTree[i-regLen]
                        edge2 = meanTree[j-regLen]
                    elif i<regLen and j>=regLen:
                        edge1 = regTree[i]
                        edge2 = meanTree[j-regLen]
                    # Calculate distance, then use (signed) correlation function to calculate
                    baseDist = getDist(edge1,edge2)
                    corrMat[i,j] = corrFunVec[baseDist+maxDist]
    else:
        regLen = len(regTree)
        corrMatSize = 2*regLen
        corrMat = np.zeros((corrMatSize,corrMatSize))
        corrFunVec = np.zeros(2*corrMatSize+1)
        for i in range(-corrMatSize,corrMatSize+1):
            corrFunVec[i+corrMatSize] = corrFun(i)
        
        for i in range(0,corrMatSize):
            for j in range(i,corrMatSize):
                if i<regLen and j<regLen:
                    baseDist = abs(i-j)
                elif i>=regLen and j>=regLen:
                    baseDist = -(abs(i-j)+1)
                elif i<regLen and j>=regLen:
                    if j-regLen<=i:
                        baseDist = -(abs(i-(j-regLen))+1)
                    else:
                        baseDist = abs(i-(j-regLen))
                
                corrMat[i,j] = corrFunVec[baseDist+corrMatSize]
                corrMat[j,i] = corrMat[i,j]
    return corrMat


def corrFunFromDist(dist,nodeVarFun):
    """
    corrFunFromDist - convert statistics of fitness landscape to correlation function
    :param dist:
    :param nodeVarFun:
    :return corrFunVal: correlation function value
    """
    corrFunVal = (nodeVarFun(abs(dist+1))-2*nodeVarFun(abs(dist))+nodeVarFun(abs(dist-1)))/2
    if dist<0:
        corrFunVal = -corrFunVal
    return corrFunVal


"""
Response function methods
"""

def conditionalVar(corrMat):
    """
    conditional variance of first entry
    :param corrMat: correlation matrix
    :return conVar: conditional variance
    """
    conVarVec = np.linalg.solve(corrMat[1:,1:],corrMat[1:,0])
    conVarDiff = np.dot(conVarVec,corrMat[1:,0])
    conVar = corrMat[0,0]-conVarDiff
    return conVar


def linearPathResponseFunction(corrMat):
    """
    Generate mean and regular edge response functions from full correlation matrix.
    Matrix structure is: first entries - regular edges, second entries - mean edges.
    Graph topology: line
    :param corrMat:
    :return meanResponse,regResponse:
    """
    responseLen = int(np.shape(corrMat)[1]/2)

    responseIdxVec = np.concatenate((np.arange(0,responseLen-1),np.arange(responseLen,2*responseLen-1)))
    responseIdxVec = responseIdxVec.astype(int)

    subCorrMat = corrMat[np.ix_(responseIdxVec,responseIdxVec)]
    subCorrVec = corrMat[responseIdxVec,2*responseLen-1]
    responseFun = np.linalg.solve(subCorrMat,subCorrVec)
    # reversal to get response function as increasing function of distance
    regResponse = responseFun[(responseLen-2)::-1]
    meanResponse = responseFun[:(responseLen-2):-1]
    return meanResponse,regResponse


def genResponseStats(corrMat,regTree,meanTree):
    """
    Generate collection of response vectors for an entire run at once. Goes through regTree in order.
    :param corrMat: Full correlation matrix. Structure is: regular edges, mean edges
    :return responseFunsMean: array of tuples of the form: ((regIndices,regResponse),(meanIndices,meanResponse)),
    where the ith such tuple corresponds to choosing the ith mean. Feeds into setValWithResponse in meanEdge class
    :return conditionalVarsMean: Conditional variances for mean edges
    :return conditionalVarsReg: Conditional variances for regular edges
    """
    # copy trees
    regTreeCopy,meanTreeCopy = copyTrees(regTree,meanTree)

    # label all trees as being unassigned
    regTreeCopy = cleanTree(regTreeCopy)
    meanTreeCopy = cleanTree(meanTreeCopy)

    # initialize mean response functions, conditional variances
    responseFunsMean = [None]*len(meanTree)
    conditionalVarsMean = [None]*len(meanTree)
    conditionalVarsReg = [None]*len(regTree)
    # Loop through regular edges to pick means in right order
    usedMeans = np.zeros(len(meanTree))
    usedMeanIdx = 0
    for i,regEdge in enumerate(regTreeCopy):
        # If mean not set, add its response
        if regEdge.baseMean.hasValue==False:
            curMeanTreeIdx = regEdge.baseMean.treeIdx
            usedMeans[usedMeanIdx] = curMeanTreeIdx
            usedMeanIdx = usedMeanIdx+1
            regEdge.baseMean.hasValue = True

            # Submatrix of corrMat. Mean of interest first, then regular edges, then past means
            curMeanIdx = [curMeanTreeIdx+len(regTree)]
            pastMeanIdx = usedMeans[range(0,usedMeanIdx-1)]+len(regTree)
            subMatIdx = np.concatenate((curMeanIdx,range(0,i),pastMeanIdx))
            subMatIdx = subMatIdx.astype(int)
            subCorrMat = corrMat[np.ix_(subMatIdx,subMatIdx)]

            # generate response function. 2 sets of tuples, for reg and mean edges.
            # tuples of form (indices, weight values)

            # indices
            meanResponseIdx = pastMeanIdx-len(regTree)
            regResponseIdx = range(0,i)
            # generate response function vector
            responseFunctionVec = gaussResponseFunction(subCorrMat)
            # split vector
            regResponseVec = responseFunctionVec[0:i]
            meanResponseVec = responseFunctionVec[i:]
            # assign
            responseFunsMean[curMeanTreeIdx] = ((regResponseIdx,regResponseVec),(meanResponseIdx,meanResponseVec))

            # generate conditional variance
            conditionalVarsMean[curMeanTreeIdx] = conditionalVar(subCorrMat)
            # Set conditional variances (in actual object, not copy!)
            meanTree[curMeanTreeIdx].edgeVar = conditionalVarsMean[curMeanTreeIdx]

        # conditional variance of regular edge

        # indexing
        pastMeanIdx = usedMeans[range(0,usedMeanIdx)]+len(regTree)
        subMatIdx = np.concatenate(([i],range(0,i),pastMeanIdx))
        subMatIdx = subMatIdx.astype(int)
        subCorrMat = corrMat[np.ix_(subMatIdx,subMatIdx)]
        # assignment
        conditionalVarsReg[i] = conditionalVar(subCorrMat)
        regTree[i].edgeVar = conditionalVarsReg[i]


    return responseFunsMean,conditionalVarsMean,conditionalVarsReg


def gaussResponseFunction(corrMat):
    """
    generate response function for first index. No substructure.
    :param corrMat: correlation matrix
    :return responseFunVec: vector for response
    """
    responseIdxVec = range(1,corrMat.shape[0])
    subCorrMat = corrMat[np.ix_(responseIdxVec,responseIdxVec)]
    subCorrVec = corrMat[responseIdxVec,0]
    responseFunVec = np.linalg.solve(subCorrMat,subCorrVec)
    return responseFunVec


"""
Single walks, no time structure
"""

def fixedTopologyWalk(regTree,meanTree,weightFun,nodeCorrFun=None,straightPath=False,firstRegEdge=0,
                      firstMeanEdge=0,responseFunsMean=None):
    """
    Random walk down a fixed topology, with given correlation function and weight function.
    Returns correlation matrix as well as the two trees.

    :param nodeCorrFun: Correlation function
    :param weightFun: Weighting function
    :param regTree: Topology of regular tree
    :param meanTree: Topology of mean tree
    :param straightPath: Indicator of whether or not walk is in single direction. If it is, use fast corrMat generator
    :param firstRegEdge: index of first regular edge which is not fixed
    :param firstMeanEdge: index of first mean edge which is not fixed
    :param responseFunsMean: Precomputed mean response function. Length the same as meanTree.
    :return corrMat,regTree,meanTree: Correlation matrix, reg tree, mean tree
    """

    if responseFunsMean==None:
        # generate correlation matrix
        def corrFun(x):
            return corrFunFromDist(x,nodeCorrFun)
        corrMat = genCorrMat(regTree,meanTree,corrFun,straightPath)
        # generate random walk
        usedMeans = np.zeros(len(meanTree))
        for i in range(0,firstMeanEdge):
            usedMeans[i] = meanTree[i].treeIdx
        usedMeanIdx = firstMeanEdge
        for i in range(firstRegEdge,len(regTree)):
            regEdge = regTree[i]
            # If mean not set, then set it
            if regEdge.baseMean.hasValue==False:
                usedMeans[usedMeanIdx] = regEdge.baseMean.treeIdx
                usedMeanIdx = usedMeanIdx+1
                # Submatrix of corrMat. Mean to generate first.
                curMeanIdx = [usedMeans[usedMeanIdx-1]+len(regTree)]
                pastMeanIdx = usedMeans[range(0,usedMeanIdx-1)]+len(regTree)
                subMatIdx = np.concatenate((curMeanIdx,range(0,i),pastMeanIdx))
                subMatIdx = subMatIdx.astype(int)
                subCorrMat = corrMat[np.ix_(subMatIdx,subMatIdx)]
                subMeanTree = [meanTree[i] for i in usedMeans[range(0,usedMeanIdx-1)].astype(int)]
                regEdge.baseMean.genMeanVal(regTree[0:i],subMeanTree,subCorrMat)
            # calculate conditional mean/variance
            pastMeanIdx = usedMeans[range(0,usedMeanIdx)]+len(regTree)
            subMatIdx = np.concatenate(([i],range(0,i),pastMeanIdx))
            subMatIdx = subMatIdx.astype(int)
            subCorrMat = corrMat[np.ix_(subMatIdx,subMatIdx)]
            varVal = conditionalVar(subCorrMat)
            meanVal = regEdge.baseMean.edgeVal
            # generate new edge value
            newRegEdgeVal = genEdgeVal(meanVal,varVal,weightFun)
            regEdge.setVal(newRegEdgeVal)

        return corrMat,regTree,meanTree
    else:
        # response function precomputed
        for i in range(firstRegEdge,len(regTree)):
            regEdge = regTree[i]
            # If mean not set, then set it
            if regEdge.baseMean.hasValue==False:
                meanTreeIdx = regEdge.baseMean.treeIdx
                regEdge.baseMean.setValWithResponse(regTree,meanTree,responseFunsMean[meanTreeIdx])
            # generate new edge value
            newRegEdgeVal = genEdgeVal(regEdge.baseMean.edgeVal,regEdge.edgeVar,weightFun)
            regEdge.setVal(newRegEdgeVal)
        return regTree,meanTree



def initializeSinglePathWalk(pathLen):
    """
    Initialize a tree down a single path
    :param pathLen: length of walk
    :return meanTree,regTree:
    """
    meanTree = [None for i in range(pathLen)]
    regTree = [None for i in range(pathLen)]

    for i,val in enumerate(meanTree):
        meanTree[i] = meanEdge(treeAddress=range(0,i),treeIdx=i)
        regTree[i] = regEdge(treeAddress=range(0,i+1),baseMean=meanTree[i],treeIdx=i)

    return meanTree,regTree


def initializeBranchedPathWalk(trunkLen,branchLen):
    """
    Initialize a tree down a single path
    :param trunkLen: length of shared path
    :param branchLen: length of branched path
    :return meanTree,regTree:
    """
    meanTree = [None]*(trunkLen+2*branchLen-1)
    regTree = [None]*(trunkLen+2*branchLen)

    for i,val in enumerate(regTree):
        if i<trunkLen:
            meanAddress = i*[0]
            regAddress = (i+1)*[0]
            meanTree[i] = meanEdge(treeAddress=meanAddress,treeIdx=i)
            meanIdx = i
        elif i<(trunkLen+branchLen):
            meanAddress = i*[0]
            regAddress = (i+1)*[0]
            meanTree[i] = meanEdge(treeAddress=meanAddress,treeIdx=i)
            meanIdx = i
        elif i==(trunkLen+branchLen):
            regAddress = trunkLen*[0]+(i+1-branchLen-trunkLen)*[1]
            meanIdx = trunkLen
        else:
            meanAddress = trunkLen*[0]+(i-branchLen-trunkLen)*[1]
            regAddress = trunkLen*[0]+(i+1-branchLen-trunkLen)*[1]
            meanTree[i-1] = meanEdge(treeAddress=meanAddress,treeIdx=(i-1))
            meanIdx = i-1


        regTree[i] = regEdge(treeAddress=regAddress,baseMean=meanTree[meanIdx],treeIdx=i)

    return meanTree,regTree


def powerLawWalk(powerExp,weightFun,walkLen,straightPath=True,regTree=None,meanTree=None,firstRegEdge=0,specAlgo=None):
    """
    Random walk, with power law correlations
    :param powerExp: exponent of correlation function
    :param weightFun: choice function
    :param walkLen: length of walk
    :param straightPath: indication of straight path, and whether or not straight path algorithm should be used
    :param regTree: regular tree
    :param meanTree: mean tree
    :param firstRegEdge: index of first non-assigned regular edge
    :param specAlgo: special walk algorithm
    :return: regular edge values, mean edge values, correlation matrix
    """
    def nodeCorrFun(x):
        return np.power(x,powerExp)
    if straightPath==True and regTree==None:
        meanTree,regTree = initializeSinglePathWalk(walkLen)
    corrMat,regTreeFinal,meanTreeFinal = \
        fixedTopologyWalk(regTree[0:walkLen],meanTree,nodeCorrFun=nodeCorrFun,weightFun=weightFun,
                          straightPath=straightPath,firstRegEdge=firstRegEdge)
    return pathVals(regTreeFinal),pathVals(meanTreeFinal),corrMat


def uphillPathExists(meanVal,varVal,numDims):
    """
    Estimation of whether or not there is an uphill path at a particular mean and variance
    :param meanVal: - mean of distribution
    :param varVal: variance
    :param numDims: number of paths to explore
    :return:
    """
    if norm.ppf(1-(1/numDims),loc=meanVal,scale=np.sqrt(varVal))>0:
        uphillExists = True
    else:
        uphillExists = False
    return uphillExists


"""
Single walks, time structure
"""


def drawFirstWaitTime(time0,time1):
    """
    Given two exponential waiting times, return the type that happened first, and its waiting time

    :param time0: Expectation of first waiting time
    :param time1: Expectation of second waiting time
    :return chosenProcess,waitTime: Returns which happened first (0 or 1), and actual time waited
    """

    # Choose which came first
    time0Prob = time1/(time0+time1)
    Z = np.random.rand()
    if Z<time0Prob:
        chosenProcess = 0
    else:
        chosenProcess = 1


    # Draw the actual waiting time

    # Expected wait
    expectedWait = time0*time1/(time1+time0)
    waitTime = np.random.exponential(expectedWait)

    return chosenProcess,waitTime


def didExpHappen(waitScale,windowSize):
    """
    Given exponential with time waitScale, did it happen in windowSize?

    :param waitScale: Expectation of waiting time
    :param windowSize: Window to wait in
    :return eventHappened,waitTime: whether or not event happened, and actual time waited
    """

    waitTime = np.random.exponential(waitScale)
    eventHappened = waitTime<windowSize

    return eventHappened,waitTime


def fixedTopologyWalkExpTimeCorr(regTree,meanTree,nodeCorrFun,weightFun,randdt,randScale,
                                 dynTimeScale="invProb",straightPath=False,firstRegEdge=0):
    """
    Random walk down a fixed topology, with given correlation function and weight function. Exponential time correlation

    Returns correlation matrix as well as the two trees.

    :param nodeCorrFun: Correlation function
    :param weightFun: Weighting function
    :param regTree: Topology of regular tree
    :param meanTree: Topology of mean tree
    :param randdt: Typical time in between randomization events
    :param randScale: Timescale of randomness; upon choice, time change is randdt*randScale
    :param dynTimeScale: Make appropriate choice of timescale for dynamics
    :param straightPath: Indicator of whether or not walk is in single direction. If it is, use fast corrMat generator
    :param firstRegEdge: index of first regular edge which is not fixed
    :return corrMat,regTree,meanTree: Correlation matrix, reg tree, mean tree
    """



    # generate correlation matrix
    def corrFun(x):
        return corrFunFromDist(x,nodeCorrFun)
    corrMat = genCorrMat(regTree,meanTree,corrFun,straightPath)



    # generate random walk

    # Initialize mean values visited
    usedMeans = np.zeros(len(meanTree))
    usedMeanIdx = 0

    # initialize time
    t = 0
    # time to next randomization event
    nextRandDt = randdt
    # initialize choice time function
    if dynTimeScale == "invProb":
        if weightFun=="uphillWalk":
            def choiceTimeFunc(meanVal,varVal):
                # Calculate P(edgeVal>0)
                choiceProb = norm.cdf(meanVal/np.sqrt(varVal))
                return 1/choiceProb
    else:
        def choiceTimeFunc(meanVal,varVal):
            return 1


    for i in range(firstRegEdge,len(regTree)):
        regEdge = regTree[i]

        if regEdge.baseMean.hasValue==False:
            # Once direction found, compute
            usedMeans[usedMeanIdx] = regEdge.baseMean.treeIdx
            usedMeanIdx = usedMeanIdx+1
            # Submatrix of corrMat. Mean to generate first.
            curMeanIdx = [usedMeans[usedMeanIdx-1]+len(regTree)]
            pastMeanIdx = usedMeans[range(0,usedMeanIdx-1)]+len(regTree)
            subMatIdx = np.concatenate((curMeanIdx,range(0,i),pastMeanIdx))
            subMatIdx = subMatIdx.astype(int)
            subCorrMat = corrMat[np.ix_(subMatIdx,subMatIdx)]
            subMeanTree = [meanTree[i] for i in usedMeans[range(0,usedMeanIdx-1)].astype(int)]
            regEdge.baseMean.genMeanVal(regTree[0:i],subMeanTree,subCorrMat)
            regEdge.baseMean.assignmentTime = t

        # Run dynamics

        directionFound = False

        # calculate conditional variance
        pastMeanIdx = usedMeans[range(0,usedMeanIdx)]+len(regTree)
        subMatIdx = np.concatenate(([i],range(0,i),pastMeanIdx))
        subMatIdx = subMatIdx.astype(int)
        subCorrMat = corrMat[np.ix_(subMatIdx,subMatIdx)]
        varVal = conditionalVar(subCorrMat)


        while directionFound==False:
            # calculate conditional mean
            meanVal = regEdge.baseMean.edgeVal
            # Calculate time to find direction
            directionFindTime = choiceTimeFunc(meanVal,varVal)

            directionFound,findingTime = didExpHappen(directionFindTime,nextRandDt)
            if directionFound==True:
                t = t+findingTime
                nextRandDt = nextRandDt-findingTime
            else:
                t = t+nextRandDt
                nextRandDt = randdt
                # generate time correlated ensemble

                # Choose past values. Copies of originals. Mean subtree should be in order explored.
                regTreeIdx = list(range(0,i))
                regSubTree,meanSubTree = copySubTree(regTree,regTreeIdx)


                # clean past values
                cleanTree(regSubTree)
                cleanTree(meanSubTree)

                # Generate new values NEED TO FIX DOES NOT MODIFY LAST MEAN
                fixedTopologyWalk(regSubTree,meanSubTree,nodeCorrFun=nodeCorrFun,weightFun="constant",
                                  straightPath=False,firstRegEdge=0)

                # Combine old values and new values appropriately
                for regIdx,regSubEdge in enumerate(regSubTree):
                    newRegEdgeVal = (np.exp(-randdt/randScale)*regTree[regIdx].edgeVal
                                     +np.sqrt(1-np.exp(-2*randdt/randScale))*regSubEdge.edgeVal)
                    regTree[i].setVal(newRegEdgeVal)

                for meanIdx,meanSubEdge in enumerate(meanSubTree):
                    newMeanEdgeVal = (np.exp(-randdt/randScale)*meanTree[usedMeans[meanIdx]].edgeVal
                                      +np.sqrt(1-np.exp(-2*randdt/randScale))*meanSubEdge.edgeVal)
                    meanTree[usedMeans[meanIdx]].setVal(newMeanEdgeVal)





        # Path found, generate new edge value
        newRegEdgeVal = genEdgeVal(meanVal,varVal,weightFun)
        regEdge.setVal(newRegEdgeVal)
        regEdge.assignmentTime = t

    return corrMat,regTree,meanTree

"""
Multiple walks, no time structure
"""


def fixedTopologyWalkMultRuns(regTree,meanTree,nodeCorrFun,weightFun,numTrials=1,straightPath=False,
                              firstRegEdge=0,firstMeanEdge=0):
    """
    Multiple runs of a random walk down a fixed topology, with given correlation function and weight function.
    Returns correlation matrix as well as the two trees.

    :param regTree: Topology of regular tree
    :param meanTree: Topology of mean tree
    :param nodeCorrFun: Correlation function
    :param weightFun: Weighting function
    :param numTrials: number of trials
    :param straightPath: Indicator of whether or not walk is in single direction. If it is, use fast corrMat generator
    :param firstRegEdge: index of first regular edge which is not fixed
    :param firstMeanEdge: index of first mean edge which hasn't been drawn
    :return regValsTotal,meanValsTotal,timeValsTotal,corrMat: Regular values observed, mean values observed, time taken
    at each step, Correlation matrix
    """

    # initialize data vectors
    regValsTotal = np.zeros((numTrials,len(regTree)))
    meanValsTotal = np.zeros((numTrials,len(meanTree)))
    timeValsTotal = np.zeros((numTrials,len(meanTree)))
    # generate correlation matrix
    def corrFun(x):
        return corrFunFromDist(x,nodeCorrFun)

    corrMat = genCorrMat(regTree,meanTree,corrFun,straightPath)

    # cutoff for using analytic approximation to normal cdf
    normCDFcutoff = 4

    # generate response functions and variances
    responseFunsMean,conditionalVarsMean,conditionalVarsReg = genResponseStats(corrMat,regTree,meanTree)
    # generate random walk
    for n in range(0,numTrials):
        usedMeans = np.zeros(len(meanTree))
        usedMeanIdx = 0

        # copy
        regTreeCopy,meanTreeCopy = copyTrees(regTree,meanTree)
        # clear indices
        meanTreeCopy = cleanTree(meanTreeCopy)
        regTreeCopy[firstRegEdge:] = cleanTree(regTreeCopy[firstRegEdge:])

        for i in range(0,firstRegEdge):
            # preset edges
            regValsTotal[n,i] = regTreeCopy[i].edgeVal
        # conduct walk with response functions
        fixedTopologyWalk(regTreeCopy,meanTreeCopy,weightFun,firstRegEdge=firstRegEdge,firstMeanEdge=firstMeanEdge,
                          responseFunsMean=responseFunsMean)
        # save data
        for i in range(firstRegEdge,len(regTree)):
            regEdge = regTreeCopy[i]
            meanTreeIdx = regEdge.baseMean.treeIdx
            meanValsTotal[n,meanTreeIdx] = regEdge.baseMean.edgeVal
            regValsTotal[n,i] = regEdge.edgeVal
            if weightFun == "uphillWalk":
                # find mean time taken
                uphillMean = regEdge.baseMean.edgeVal
                uphillStdDev = np.sqrt(regEdge.baseMean.edgeVar)
                effectiveMean = uphillMean/uphillStdDev
                if -effectiveMean<normCDFcutoff:
                    meanTime = np.power(norm.cdf(effectiveMean),-1)
                else:
                    meanTime = np.sqrt(2*np.pi)*np.abs(effectiveMean)*np.exp(np.power(effectiveMean,2)/2)
                timeValsTotal[n,i] = meanTime*np.random.exponential()
            else:
                timeValsTotal[n,i] = 0

    return regValsTotal,meanValsTotal,timeValsTotal,corrMat


def powerLawWalkMultRun(powerExp,weightFun,walkLen,numTrials=1,straightPath=True,regTree=None,meanTree=None,firstRegEdge=0,specAlgo=None):
    """
    Random walk, with power law correlations
    :param powerExp: exponent of correlation function
    :param weightFun: choice function
    :param walkLen: length of walk
    :param numTrials: number of trials to run
    :param straightPath: indication of straight path, and whether or not straight path algorithm should be used
    :param regTree: regular tree
    :param meanTree: mean tree
    :param firstRegEdge: index of first non-assigned regular edge
    :param specAlgo: special walk algorithm
    :return: regular edge values, mean edge values, correlation matrix
    """
    def nodeCorrFun(x):
        return np.power(x,powerExp)
    if straightPath==True and regTree==None:
        meanTree,regTree = initializeSinglePathWalk(walkLen)
    return fixedTopologyWalkMultRuns(regTree,meanTree,nodeCorrFun,weightFun,numTrials,straightPath,firstRegEdge)


def branchingWalkMultRun(trunkLen,branchLen,nodeCorrFun,weightFun,numTrials=1):
    meanTree,regTree = initializeBranchedPathWalk(trunkLen,branchLen)
    return fixedTopologyWalkMultRuns(regTree,meanTree,nodeCorrFun,weightFun,numTrials=numTrials,
                                     straightPath=False,firstRegEdge=0)


def powerLawBranchingWalkMultRun(powerExp,weightFun,trunkLen,branchLen,numTrials=1,specAlgo=None):
    meanTree,regTree = initializeBranchedPathWalk(trunkLen,branchLen)
    return powerLawWalkMultRun(powerExp,weightFun,walkLen=None,numTrials=numTrials,straightPath=False,
                               regTree=regTree,meanTree=meanTree,firstRegEdge=0,specAlgo=None)


"""
Multiple walks, time structure
"""

# Note: currently only supports straight path walks. Extend to work with non-straight path walks?
def fixedTopologyWalkMultRunsExpTimeCorr(regTree,meanTree,nodeCorrFun,weightFun,randdt,
                                         randScale,dynTimeScale="invProb",numTrials=1,
                                         straightPath=False,firstRegEdge=0,firstMeanEdge=0):
    """
    Multiple runs of a random walk down a fixed topology, with given correlation function and weight function.
    Returns correlation matrix as well as the two trees.

    :param regTree: Topology of regular tree
    :param meanTree: Topology of mean tree
    :param nodeCorrFun: Correlation function
    :param weightFun: Weighting function
    :param randdt: time between randomization events
    :param randScale: Timescale of exponential decay of correlations
    :param dynTimeScale: Method of computing waiting time for a step
    :param numTrials: number of trials
    :param straightPath: Indicator of whether or not walk is in single direction. If it is, use fast corrMat generator
    :param firstRegEdge: index of first regular edge which is not fixed
    :return regValsTotal,meanValsTotal,timeValsTotal,corrMat: Regular values observed, mean values observed,
            assignment times observed, Correlation matrix
    """

    # initialize data vectors
    regValsTotal = np.zeros((numTrials,len(regTree)))
    meanValsTotal = np.zeros((numTrials,len(meanTree)))
    timeValsTotal = np.zeros((numTrials,len(regTree)))
    # generate correlation matrix
    def corrFun(x):
        return corrFunFromDist(x,nodeCorrFun)
    corrMat = genCorrMat(regTree,meanTree,corrFun,straightPath)
    import time
    t = time.time()
    # generate response functions and variances
    responseFunsMean,conditionalVarsMean,conditionalVarsReg = genResponseStats(corrMat,regTree,meanTree)
    # generate random walk
    for n in range(0,numTrials):

        # copy
        regTreeCopy,meanTreeCopy = copyTrees(regTree,meanTree)
        # Set used means
        usedMeans = np.zeros(len(meanTree))
        for i in range(0,firstMeanEdge):
            usedMeans[i] = meanTreeCopy[i].treeIdx
            meanValsTotal[n,i] = meanTreeCopy[i].edgeVal
        usedMeanIdx = firstMeanEdge
        # clear indices
        meanTreeCopy[firstMeanEdge:] = cleanTree(meanTreeCopy[firstMeanEdge:])
        regTreeCopy[firstRegEdge:] = cleanTree(regTreeCopy[firstRegEdge:])

        for i in range(0,firstRegEdge):
            # preset edges
            regValsTotal[n,i] = regTreeCopy[i].edgeVal
        # Run dynamics
        # initialize time
        t = 0
        # time to next randomization event
        nextRandDt = randdt
        # initialize choice time function
        if dynTimeScale == "invProb":
            if weightFun=="uphillWalk":
                def choiceTimeFunc(meanVal,varVal):
                    # Calculate P(edgeVal>0)
                    choiceProb = norm.cdf(meanVal/np.sqrt(varVal))
                    return 1/choiceProb
        else:
            def choiceTimeFunc(meanVal,varVal):
                return 1

        for i in range(firstRegEdge,len(regTree)):
            regEdge = regTreeCopy[i]
            # If mean not set, then set it
            if regEdge.baseMean.hasValue == False:
                meanTreeIdx = regEdge.baseMean.treeIdx
                usedMeans[usedMeanIdx] = meanTreeIdx
                usedMeanIdx = usedMeanIdx+1
                regEdge.baseMean.setValWithResponse(regTreeCopy,meanTreeCopy,responseFunsMean[meanTreeIdx])
                meanValsTotal[n,meanTreeIdx] = regEdge.baseMean.edgeVal
            # calculate conditional mean
            meanVal = regEdge.baseMean.edgeVal
            # Calculate time to find direction
            directionFindTime = choiceTimeFunc(meanVal,conditionalVarsReg[i])
            directionFound = False
            meanAdjust = 0 # value to be added via time dependence
            totalTimeSpent = 0 # time spent finding an edge
            # real time dynamics
            while directionFound == False:
                # Check if it happened before randomization
                directionFound,findingTime = didExpHappen(directionFindTime,nextRandDt)
                if directionFound==True:
                    t = t+findingTime
                    nextRandDt = nextRandDt-findingTime
                    if straightPath:
                        oldWeight = np.exp(-totalTimeSpent/randScale)
                        newWeight = np.sqrt(1-np.exp(-2*totalTimeSpent/randScale))
                        if i>0:
                            # draw the rest of the path
                            subTreeIdx = list(range(0,i))
                            regSubTree,meanSubTree = copySubTree(regTree,subTreeIdx)
                            meanSubTree[0].setVal(meanAdjust)
                            fixedTopologyWalk(regSubTree,meanSubTree,weightFun="constant",
                                              straightPath=True,responseFunsMean=responseFunsMean)
                            initMeanCopy = copy(regEdge.baseMean)
                            initMeanCopy.setValWithResponse(regSubTree,meanSubTree,responseFunsMean[i])
                            # reverse, change signs, and add to original path
                            for j in range(0,i):
                                # reversal, with sign change
                                newRegEdgeVal = oldWeight*regTreeCopy[j].edgeVal-newWeight*regSubTree[i-j-1].edgeVal
                                regTreeCopy[j].setVal(newRegEdgeVal)
                                if j>0:
                                    # reversal, no sign change
                                    newMeanEdgeVal = oldWeight*meanTreeCopy[j].edgeVal+newWeight*meanSubTree[i-j].edgeVal
                                    meanTreeCopy[j].setVal(newMeanEdgeVal)
                                else:
                                    # Furthest in past value
                                    newInitVal = oldWeight*meanTreeCopy[0].edgeVal+newWeight*initMeanCopy.edgeVal
                                    meanTreeCopy[0].setVal(newInitVal)
                        # Update current edge
                        regEdge.baseMean.setVal(meanVal)

                else:
                    # random dynamics
                    t = t+nextRandDt
                    nextRandDt = randdt
                    # generate time correlated ensemble
                    if straightPath:
                        meanAdjust = meanAdjust*+np.random.normal(scale=np.sqrt(regEdge.baseMean.edgeVar))
                        meanVal = np.exp(-randdt/randScale)*meanVal+np.sqrt(1-np.exp(-2*randdt/randScale))*meanAdjust
                        directionFindTime = choiceTimeFunc(meanVal,conditionalVarsReg[i])
                        totalTimeSpent = totalTimeSpent+randdt
                    else:
                        # Choose past values. Copies of originals.
                        # clean past values
                        subTreeIdx = list(range(0,i))
                        regSubTree,meanSubTree = copySubTree(regTree,subTreeIdx)

                        # Generate new values
                        # conduct walk with response functions NEEDS FIXING DOES NOT MODIFY CURRENT MEAN
                        fixedTopologyWalk(regSubTree,meanSubTree,weightFun="constant",firstRegEdge=0,
                                          responseFunsMean=responseFunsMean)

                        # Combine old values and new values appropriately
                        for regIdx,regSubEdge in enumerate(regSubTree):
                            newRegEdgeVal = np.exp(-randdt/randScale)*regTreeCopy[regIdx].edgeVal+np.sqrt(
                                1-np.exp(-2*randdt/randScale))*regSubEdge.edgeVal
                            regTreeCopy[regIdx].setVal(newRegEdgeVal)

                        for meanIdx,meanSubEdge in enumerate(meanSubTree):
                            newMeanEdgeVal = (np.exp(-randdt/randScale)*meanTreeCopy[int(usedMeans[meanIdx])].edgeVal
                                              +np.sqrt(1-np.exp(-2*randdt/randScale))*meanSubEdge.edgeVal)
                            meanTreeCopy[int(usedMeans[meanIdx])].setVal(newMeanEdgeVal)

            # generate new edge value
            newRegEdgeVal = genEdgeVal(regEdge.baseMean.edgeVal,regEdge.edgeVar,weightFun)
            regEdge.setVal(newRegEdgeVal)
            regEdge.assignmentTime = t
            regValsTotal[n,i] = regEdge.edgeVal
            timeValsTotal[n,i] = regEdge.assignmentTime

    return regValsTotal,meanValsTotal,timeValsTotal,corrMat


def powerLawWalkMultRunTimeCorr(powerExp,weightFun,randdt,randScale,
                                walkLen,dynTimeScale="invProb",numTrials=1,straightPath=True,regTree=None,
                                meanTree=None,firstRegEdge=0,firstMeanEdge=0,specAlgo=None,firstMeanVals=None):
    """
    Random walk, with power law correlations
    :param powerExp: exponent of correlation function
    :param weightFun: choice function
    :param walkLen: length of walk
    :param numTrials: number of trials to run
    :param straightPath: indication of straight path, and whether or not straight path algorithm should be used
    :param regTree: regular tree
    :param meanTree: mean tree
    :param firstRegEdge: index of first non-assigned regular edge
    :param specAlgo: special walk algorithm
    :return: regular edge values, mean edge values, assigment times, correlation matrix
    """
    def nodeCorrFun(x):
        return np.power(x,powerExp)
    if straightPath==True and regTree==None:
        meanTree,regTree = initializeSinglePathWalk(walkLen)
        if firstMeanVals!=None:
            firstMeanEdge = len(firstMeanVals)
            for meanIdx,meanVal in enumerate(firstMeanVals):
                meanTree[meanIdx].setVal(meanVal)

    return fixedTopologyWalkMultRunsExpTimeCorr(regTree,meanTree,nodeCorrFun,weightFun,randdt,randScale,
                                                dynTimeScale=dynTimeScale,numTrials=numTrials,straightPath=straightPath,
                                                firstRegEdge=firstRegEdge,firstMeanEdge=firstMeanEdge)
