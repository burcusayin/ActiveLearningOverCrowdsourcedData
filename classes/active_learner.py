#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:18:56 2019

@author: burcusyn
"""
import sys
sys.path.append('./libact/')
import numpy as np
import copy
import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import multi_argmax
from utils import VoteAggretagor
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import multiprocessing

class ActiveLearner:
    '''This is the base class for active learning models'''

    def __init__(self, dataset, name, model, batchSize, voteAggTechnique):
        '''input: dataset -- an object of class Dataset or any inheriting classes
                  name -- name of the method for saving the results later'''
        
        self.dataset = dataset
        self.batchSize = batchSize
        self.voteAggMethod = voteAggTechnique
        self.indicesKnown = dataset.indicesKnown
        self.indicesUnknown = dataset.indicesUnknown
        self.indicesTrained = dataset.indicesKnown
        self.indicesUntrained = dataset.indicesUnknown
        self.trainData = copy.deepcopy(dataset.poolData[self.indicesKnown,:])
        self.trainLabels = copy.deepcopy(dataset.poolGoldLabels[self.indicesKnown])
        self.poolData = dataset.poolData
        self.poolGoldLabels = dataset.poolGoldLabels
        self.poolWorkerResponses = copy.deepcopy(dataset.poolWorkerResponses)
        self.poolWorkers = copy.deepcopy(dataset.poolWorkers)
        self.queriedVoteHistory = []
        self.queriedVoteOwnersHistory = []
        for i in range(len(dataset.poolData)):
            if i in self.indicesTrained:
                self.queriedVoteHistory.append([dataset.poolGoldLabels[i]])
                self.queriedVoteOwnersHistory.append([self.dataset.poolWorkers[i]])
            else:
                self.queriedVoteHistory.append([])
                self.queriedVoteOwnersHistory.append([])
                
        self.queriedVoteHistory = np.array(self.queriedVoteHistory)
                
        # base classification model
        self.model = model
        self.name = name
        self.aggregator = VoteAggretagor()
               
    def reset(self):
        
        '''forget all the points sampled by active learning and reset all sets to default of the dataset'''
        self.indicesKnown = self.dataset.indicesKnown
        self.indicesUnknown = self.dataset.indicesUnknown
        self.indicesTrained = self.dataset.indicesKnown
        self.indicesUntrained = self.dataset.indicesUnknown
        self.trainData = copy.deepcopy(self.dataset.poolData[self.dataset.indicesKnown,:])
        self.trainLabels = copy.deepcopy(self.dataset.poolGoldLabels[self.dataset.indicesKnown])
        self.poolData = self.dataset.poolData
        self.poolGoldLabels = self.dataset.poolGoldLabels
        self.poolWorkerResponses = copy.deepcopy(self.dataset.poolWorkerResponses)
        self.poolWorkers = copy.deepcopy(self.dataset.poolWorkers)
        self.queriedVoteHistory = []
        self.queriedVoteOwnersHistory = []
        for i in range(len(self.dataset.poolData)):
            if i in self.dataset.indicesKnown:
                self.queriedVoteHistory.append([self.dataset.poolGoldLabels[i]])
                self.queriedVoteOwnersHistory.append([self.dataset.poolWorkers[i]])
            else:
                self.queriedVoteHistory.append([])
                self.queriedVoteOwnersHistory.append([])
        self.queriedVoteHistory = np.array(self.queriedVoteHistory)
        
    def train(self):
        
        '''train the base classification model on currently available datapoints'''
        trainDataKnown = self.trainData
        trainLabelsKnown = self.trainLabels
        trainLabelsKnown = np.ravel(trainLabelsKnown)
        self.model = self.model.fit(trainDataKnown, trainLabelsKnown)
        
    def update(self, datapoints, maxVoteCount):
        for datapoint in datapoints:
            responses = self.poolWorkerResponses[datapoint]
            selectedVoteIndex = random.choice(range(len(responses)))
            selectedVote = responses[selectedVoteIndex]
            selectedWorker = self.poolWorkers[datapoint][selectedVoteIndex]
            self.queriedVoteHistory[datapoint] = np.append(self.queriedVoteHistory[datapoint], np.array([selectedVote])).astype(int) 
            self.queriedVoteOwnersHistory[datapoint] = np.append(self.queriedVoteOwnersHistory[datapoint], np.array([selectedWorker]))
            ''' Has this datapoint already been in the training set? If yes, update its label. If not, add the datapoint to training set.'''
            if datapoint not in self.indicesTrained:
                self.indicesTrained = np.concatenate(([self.indicesTrained, np.array([datapoint])]))
                self.trainData =  np.concatenate(([self.trainData, np.array([self.poolData[datapoint]])]))
                self.trainLabels = np.concatenate(([self.trainLabels, np.array([selectedVote])]))
            else:
                if self.voteAggMethod == 'majorityVoting':
                    selectedVote = self.aggregator.majorityVoting(self.queriedVoteHistory, datapoint)
                elif self.voteAggMethod == 'DS':
                    selectedVote = self.aggregator.DS(self.queriedVoteHistory, self.queriedVoteOwnersHistory, datapoint)
                else:
                    print("no valid technique is selected, default vote aggregation technique -majotityVoting- is used!")
                    selectedVote = self.aggregator.majorityVoting(self.queriedVoteHistory, datapoint)
                self.trainLabels[np.where(self.indicesTrained == datapoint)] = selectedVote
            
            if len(self.queriedVoteHistory[datapoint]) == maxVoteCount:
                self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([datapoint])]))
                self.indicesUnknown = np.delete(self.indicesUnknown, np.where(self.indicesUnknown == datapoint))
        
    '''We donot have any test data, we are testing on the same pool.'''
    def evaluate(self, performanceMeasures):
        
        '''evaluate the performance of current classification for a given set of performance measures
        input: performanceMeasures -- a list of performance measure that we would like to estimate. Possible values are 'accuracy', 'TN', 'TP', 'FN', 'FP', 'auc' 
        output: performance -- a dictionary with performanceMeasures as keys and values consisting of lists with values of performace measure at all iterations of the algorithm'''
        performance = {}
        
        performance['numOfTrainedItems'] = len(self.indicesTrained)
        
        test_prediction1 = self.model.predict(self.dataset.poolData)  
        m1 = metrics.confusion_matrix(self.dataset.poolGoldLabels,test_prediction1)
        
        test_prd = self.model.predict(self.dataset.poolData[self.indicesUntrained]) 
        test_prediction2 = [self.trainLabels[np.where(self.indicesTrained == index)] if index in self.indicesTrained else test_prd[self.indicesUntrained == index] for index in range(len(self.dataset.poolData))]
        m2 = metrics.confusion_matrix(self.dataset.poolGoldLabels,test_prediction2)
        
        # measure accuracy
        performance['accuracy1'] = metrics.accuracy_score(self.dataset.poolGoldLabels,test_prediction1)
        performance['accuracy2'] = metrics.accuracy_score(self.dataset.poolGoldLabels,test_prediction2)
            
        # measure TP, TN, FP, FN values
        performance['TN1'] = m1[0,0]
        performance['TN2'] = m2[0,0]
     
        performance['FN1'] = m1[1,0]
        performance['FN2'] = m2[1,0]
        
        performance['TP1'] = m1[1,1]
        performance['TP2'] = m2[1,1]
        
        performance['FP1'] = m1[0,1]
        performance['FP2'] = m2[0,1]
            
        # measure fbeta scores
        performance['fbeta11'] = metrics.fbeta_score(self.dataset.poolGoldLabels, test_prediction1, average=None, beta=1)
        performance['fbeta12'] = metrics.fbeta_score(self.dataset.poolGoldLabels, test_prediction2, average=None, beta=1)
        performance['fbeta31'] = metrics.fbeta_score(self.dataset.poolGoldLabels, test_prediction1, average=None, beta=3)
        performance['fbeta32'] = metrics.fbeta_score(self.dataset.poolGoldLabels, test_prediction2, average=None, beta=3)
            
        # measure auc scores
        test_prediction1 = self.model.predict_proba(self.dataset.poolData)  
        test_prediction1 = test_prediction1[:,1]
        performance['auc1'] = metrics.roc_auc_score(self.dataset.poolGoldLabels, test_prediction1)
            
        test_prd2 = self.model.predict_proba(self.dataset.poolData[self.indicesUntrained])  
        test_prd2 = test_prd2[:,1]
        test_prediction2 = [self.trainLabels[np.where(self.indicesTrained == index)] if index in self.indicesTrained else test_prd2[self.indicesUntrained == index] for index in range(len(self.dataset.poolData))]

        performance['auc1'] = metrics.roc_auc_score(self.dataset.poolGoldLabels, test_prediction1)
        performance['auc2'] = metrics.roc_auc_score(self.dataset.poolGoldLabels, test_prediction2)
            
        return performance
    
    def evaluateOnTestSet(self, performanceMeasures):
        
        performance = {}
        
        performance['numOfTrainedItems'] = len(self.indicesTrained)
        
        test_prediction = self.model.predict(self.testData)  
        m1 = metrics.confusion_matrix(self.testLabels,test_prediction)
                
        # measure accuracy
        performance['accuracy1'] = metrics.accuracy_score(self.testLabels,test_prediction)
        
        # measure TP, TN, FP, FN values
        performance['TN1'] = m1[0,0]
        performance['FN1'] = m1[1,0]       
        performance['TP1'] = m1[1,1]       
        performance['FP1'] = m1[0,1]
            
        # measure fbeta scores
        performance['fbeta11'] = metrics.fbeta_score(self.testLabels, test_prediction, average=None, beta=1)
        performance['fbeta31'] = metrics.fbeta_score(self.testLabels, test_prediction, average=None, beta=3)
            
        # measure auc scores
        test_prediction = self.model.predict_proba(self.testData)  
        test_prediction = test_prediction[:,1]
        performance['auc1'] = metrics.roc_auc_score(self.testLabels, test_prediction)
            
        return performance
          
class ActiveLearnerRandom(ActiveLearner):
    '''Randomly samples the points'''
    
    def selectNext(self):
        query_idx = random.sample(range(len(self.indicesUnknown)), self.batchSize)
        selectedIndex = self.indicesUnknown[query_idx]
        return selectedIndex
        
class ActiveLearnerUncertainty(ActiveLearner):
    '''Points are sampled according to uncertainty sampling criterion'''
    
    def selectNext(self):
        unknownPrediction = self.model.predict_proba(self.poolData[self.indicesUnknown,:])[:,0]
        selectedIndex1toN = np.argsort(np.absolute(unknownPrediction-0.5))[:self.batchSize]
        selectedIndex = self.indicesUnknown[selectedIndex1toN]
        return selectedIndex

class ActiveLearnerCertainty(ActiveLearner):
    '''Points are sampled according to certainty sampling criterion'''
    
    def selectNext(self):
        unknownPrediction = self.model.predict_proba(self.poolData[self.indicesUnknown,:])[:,0]
        selectedIndex1toN = np.argsort(np.absolute(unknownPrediction-0.5))[-self.batchSize:]
        selectedIndex = self.indicesUnknown[selectedIndex1toN]
        return selectedIndex
    
class ActiveLearnerBlockCertainty(ActiveLearner):
    
    def __init__(self, dataset, name, model, batchSize, K, voteAggTechnique):
        
        ActiveLearner.__init__(self, dataset, name, model, batchSize, voteAggTechnique)
        self.K = K
    
    def selectNext(self):
        proba = self.model.predict_proba(self.poolData[self.indicesUnknown,:])
        proba_in = proba[:, 1]
        proba_out = proba[:, 0]
        outCount = int((self.K / (1 + self.K)) * self.batchSize)
        argMaxIn = multi_argmax(proba_in, n_instances=self.batchSize-outCount)    
        argMaxOut = multi_argmax(proba_out, n_instances=outCount)
        query_idx = np.concatenate([argMaxIn,argMaxOut])
        selectedIndex = self.indicesUnknown[query_idx]
        return selectedIndex

class ActiveLearnerLAL(ActiveLearner):
    '''Points are sampled according to a method described in K. Konyushkova, R. Sznitman, P. Fua 'Learning Active Learning from data'  '''
    
    def __init__(self, dataset, name, model, batchSize, lalModel, voteAggTechnique):
        
        ActiveLearner.__init__(self, dataset, name, model, batchSize, voteAggTechnique)
        self.model = RandomForestClassifier(class_weight='balanced', random_state=2020, oob_score=True) # bu strateji rf classifier ile calisiyor orjinalinde
        self.lalModel = lalModel
      
    def selectNext(self):
        unknown_data = self.poolData[self.indicesUnknown,:]
        known_labels = self.trainLabels
        n_lablled = np.size(self.indicesKnown)
        n_dim = np.shape(self.poolData)[1]
        
        # predictions of the trees
        temp = np.array([tree.predict_proba(unknown_data)[:,0] for tree in self.model.estimators_])
        # - average and standard deviation of the predicted scores
        f_1 = np.mean(temp, axis=0)
        f_2 = np.std(temp, axis=0)
        # - proportion of positive points
        f_3 = (sum(known_labels>0)/n_lablled)*np.ones_like(f_1)
        # the score estimated on out of bag estimate
        f_4 = self.model.oob_score_*np.ones_like(f_1)
        # - coeficient of variance of feature importance
        f_5 = np.std(self.model.feature_importances_/n_dim)*np.ones_like(f_1)
        # - estimate variance of forest by looking at avergae of variance of some predictions
        f_6 = np.mean(f_2, axis=0)*np.ones_like(f_1)
        # - compute the average depth of the trees in the forest
        f_7 = np.mean(np.array([tree.tree_.max_depth for tree in self.model.estimators_]))*np.ones_like(f_1)
        # - number of already labelled datapoints
        f_8 = np.size(self.indicesTrained)*np.ones_like(f_1)
        
        # all the featrues put together for regressor
        LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
        LALfeatures = np.transpose(LALfeatures)
            
        # predict the expercted reduction in the error by adding the point
        LALprediction = self.lalModel.predict(LALfeatures)
        # select the datapoint with the biggest reduction in the error
        selectedIndex1toN = multi_argmax(LALprediction, n_instances=self.batchSize)
        # retrieve the real index of the selected datapoint    
        selectedIndex = self.indicesUnknown[selectedIndex1toN]
        return selectedIndex
    
class ActiveLearnerQUIRE(ActiveLearner):
    """Select an action to take according to Querying Informative and Representative Examples (QUIRE) strategy.

    Query the most informative and representative examples where the metrics
    measuring and combining are done using min-max approach.

    Parameters
    ----------
    lambda: float, optional (default=1.0)
        A regularization parameter used in the regularization learning
        framework.

    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.

    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.

    References  
    ----------
    .. [1] S.-J. Huang, R. Jin, and Z.-H. Zhou. Active learning by querying
           informative and representative examples.
    """
    def __init__(self, dataset, name, model, batchSize, voteAggTechnique, lmbda, gmma, kernel, coef0, degree):
        ActiveLearner.__init__(self, dataset, name, model, batchSize, voteAggTechnique)
        self.Uindex = copy.deepcopy(self.indicesUnknown).tolist()
        self.Lindex = copy.deepcopy(self.indicesKnown).tolist()
        self.lmbda = lmbda
        X = self.poolData
        self.y = self.poolGoldLabels
        self.kernel = kernel
        if self.kernel == 'rbf':
            self.K = rbf_kernel(X=X, Y=X, gamma=gmma)
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X,
                                       Y=X,
                                       coef0=coef0,
                                       degree=degree,
                                       gamma=gmma)
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        if not isinstance(self.K, np.ndarray):
            raise TypeError('K should be an ndarray')
        if self.K.shape != (len(X), len(X)):
            raise ValueError(
                'kernel should have size (%d, %d)' % (len(X), len(X)))
        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))
        
    def selectNext(self):
        L = self.L
        Lindex = copy.deepcopy(self.indicesTrained).tolist()
        Uindex = copy.deepcopy(self.indicesUnknown).tolist()
        queryIndex = -1
        evals = {}
        y_labeled = self.trainLabels
        det_Laa = np.linalg.det(L[np.ix_(Uindex, Uindex)])
        
        # efficient computation of inv(Laa)
        M3 = np.dot(self.K[np.ix_(Uindex, Lindex)],
                    np.linalg.inv(self.lmbda * np.eye(len(Lindex))))
        M2 = np.dot(M3, self.K[np.ix_(Lindex, Uindex)])
        M1 = self.lmbda * np.eye(len(Uindex)) + self.K[np.ix_(Uindex, Uindex)]
        inv_Laa = M1 - M2
        iList = list(range(len(Uindex)))
        if len(iList) == 1:
            return Uindex[0]
        for i, each_index in enumerate(Uindex):
            # go through all unlabeled instances and compute their evaluation
            # values one by one
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            iList_r = iList[:]
            iList_r.remove(i)
            inv_Luu = inv_Laa[np.ix_(iList_r, iList_r)] - 1 / inv_Laa[i, i] * \
                np.dot(inv_Laa[iList_r, i], inv_Laa[iList_r, i].T)
            tmp = np.dot(
                L[each_index][Lindex] -
                np.dot(
                    np.dot(
                        L[each_index][Uindex_r],
                        inv_Luu
                    ),
                    L[np.ix_(Uindex_r, Lindex)]
                ),
                y_labeled,
            )
            eva = L[each_index][each_index] - \
                det_Laa / L[each_index][each_index] + 2 * np.abs(tmp)
            
            evals[each_index] = eva
        sortedEvals = [[k, v] for k, v in sorted(evals.items(), key=lambda item: item[1])]
        queryIndex = np.asarray([sortedEvals[i][0] for i in range(self.batchSize)])
        unknownSorted = np.argsort(self.indicesUnknown)
        indices = np.searchsorted(self.indicesUnknown[unknownSorted], queryIndex)
        selectedIndex = self.indicesUnknown[unknownSorted[indices]]
        return selectedIndex
      
    
class ActiveLearnerMinExpError(ActiveLearner):
    
    def selectNext(self):
        proba_in = np.ones(self.poolData[self.indicesUnknown].shape[0])
        proba_in *= self.model.predict_proba(self.poolData[self.indicesUnknown])[:, 1]
    
        predicted = []
        predictedFalse = []
        for p in proba_in:
            if p < 0.5:
                predicted.append(0)   # if magically we know that the predicted label l is true, we can add it to training set and train
                predictedFalse.append(1)  # if magically we know that the predicted label l is wrong, we add 1-l to training set and train
            else:
                predicted.append(1)
                predictedFalse.append(0)
    
        #so now think every case and train the classifier for each item separately to calculate expected error of classifier
        eRightList = []
        eWrongList = []
        for i in range(len(predicted)):
            if i in self.indicesUntrained:
                X_train = np.append(self.trainData, [self.poolData[i]], axis=0)
                y_trainRight = np.append(self.trainLabels, predicted[i])
                y_trainWrong = np.append(self.trainLabels, predictedFalse[i])
            else:
                X_train = copy.deepcopy(self.trainData)
                if self.voteAggMethod == 'majorityVoting':
                    selectedVote = self.aggregator.majorityVoting(self.queriedVoteHistory, i)
                elif self.voteAggMethod == 'DS':
                    selectedVote = self.aggregator.DS(self.queriedVoteHistory, self.queriedVoteOwnersHistory, i)
                else:
                    print("no valid technique is selected, default vote aggregation technique -majotityVoting- is used!")
                    selectedVote = self.aggregator.majorityVoting(self.queriedVoteHistory, i)
                y_trainRight = copy.deepcopy(self.trainLabels)
                y_trainWrong = copy.deepcopy(self.trainLabels)
                y_trainRight[np.where(self.indicesTrained == i)] = selectedVote
                y_trainWrong[np.where(self.indicesTrained == i)] = selectedVote
            self.model.fit(X_train, y_trainRight)
            scores = cross_val_score(self.model, X_train, y_trainRight, cv=5, scoring='f1')
            eRightList.append(scores.mean())
            self.model.fit(X_train, y_trainWrong)
            scores = cross_val_score(self.model, X_train, y_trainWrong, cv=5, scoring='f1')
            eWrongList.append(scores.mean())

    
        # update the active learner back
        self.model.fit(self.trainData, self.trainLabels)
    
        # creating bootstaps
        bootstraps = []
        for i in range(10):
            size = self.trainData.shape[0]
            bootstrap_idx = np.random.choice(range(size), size, replace=True)
            bst_x = self.trainData[bootstrap_idx, :]
            bst_y = self.trainLabels[bootstrap_idx]
            bootstraps.append([bst_x, bst_y])
    
        #defining ML classifiers to use in bootstrapped training sets
        clfs = [
            LogisticRegression(C=1., solver='lbfgs'),
            CalibratedClassifierCV(SVC(kernel="linear", C=0.025), cv=3),
            CalibratedClassifierCV(SVC(gamma=2, C=1), cv=3),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            CalibratedClassifierCV(DecisionTreeClassifier(max_depth=40), cv=3),
            CalibratedClassifierCV(RandomForestClassifier(max_depth=5, n_estimators=10), cv=3),
            CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=0.1), cv=3),
            AdaBoostClassifier(),
            GaussianNB(),
            CalibratedClassifierCV(RandomForestClassifier(n_estimators=50, max_depth=40), cv=3)]
    
        #apply multiprocessing to learn predicted label of each item in the pool by each ML classifier
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(10):
            p = multiprocessing.Process(target=self.bootstrap_computation, args=(i, clfs[i], bootstraps[i], return_dict))
            jobs.append(p)
            p.start()
    
        for proc in jobs:
            proc.join()
    
        predictions = []
        for key, value in return_dict.items():
            predictions.append(value)
    
        # calculate p_u values for each item
        p_u = []
        for i in range(len(predicted)):
            p_i = 0
            for j in range(len(predictions)):
                if predicted[i] == predictions[j][i]:
                    p_i += 1
            p_u.append(p_i / 10)
    
        # calculate expected error of each item in the pool
        expErrorList = []
        for i in range(len(p_u)):
            expErrorList.append(eWrongList[i] - p_u[i] * (eWrongList[i] - eRightList[i]))
    
        # get the indexes of min n_instances
        expErrorList = np.array(expErrorList)
        idx = np.argsort(expErrorList)
        query_idx = idx[0:self.batchSize]
        selectedIndex = self.indicesUnknown[query_idx]
        # return the selected items that have minimum expected error
        return selectedIndex
    
    def bootstrap_computation(self, i, clf, bootstrap, return_dict):
        clf.fit(bootstrap[0], bootstrap[1])
        proba_in = np.ones(self.poolData.shape[0])
        proba_in *= clf.predict_proba(self.poolData)[:, 1]
        prd = []
        for p in proba_in:
            if p < 0.5:
                prd.append(0)
            else:
                prd.append(1)
        return_dict[i] = prd