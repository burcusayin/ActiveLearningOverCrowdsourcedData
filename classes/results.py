#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:23:42 2019

@author: burcusyn
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


class Results:
    '''The class that saves, loads and plots the results.'''
    
    def __init__(self, experiment = None, nExperiments = None):
        
        # the performances measures that can be computed
        self.existingMetrics = ['accuracy', 'auc', 'IoU', 'dice', 'f-measure', 'f1', 'f3']
        
        if experiment is not None:
            experiment.dtstname = experiment.dataset.__class__.__name__
            self.nIterations = experiment.nIterations
            self.performanceMeasures = experiment.performanceMeasures
            self.dataset = experiment.dataset
            self.alearners = []
            for alearner in experiment.alearners:
                self.alearners.append(alearner.name)
            self.comment = experiment.comment
            self.nExperiments = nExperiments
            
            self.performances = dict()
            for alearner in self.alearners:
                self.performances[alearner] = dict()
                for performanceMeasure in self.performanceMeasures:
                    self.performances[alearner][performanceMeasure] = np.array([[]])
        
    
    def addPerformance(self, performance):
        '''This function adds performance measures of new experiments'''
        for alearner in performance:
            for performanceMeasure in performance[alearner]:
                if np.size(self.performances[alearner][performanceMeasure])==0:
                    self.performances[alearner][performanceMeasure] = np.array([performance[alearner][performanceMeasure]])
                else:
                    self.performances[alearner][performanceMeasure] = np.concatenate((self.performances[alearner][performanceMeasure], np.array([performance[alearner][performanceMeasure]])), axis=0)
                    
                    
    def saveResults(self, filename):
        '''Save the current results to a file filename in ./exp folder'''
        state = self.__dict__.copy()
        pkl.dump(state, open( './exp/'+filename+'.p', "wb" ) )   
        
    
    def readResult(self, filename):
        '''Read the results from filename from ./exp folder'''
        state = pkl.load( open ('./exp/'+filename+'.p', "rb") )
        self.__dict__.update(state)
        
    def mergeResults(self, file1, file2, dest):
        '''Merge two result files into one'''
        
        #To load from pickle file
        data = []
        with open('./exp/'+file1+'.p', 'rb') as f:
            try:
                while True:
                    data.append(pkl.load(f))
            except EOFError:
                pass
            
        print("data after first file:", len(data))
        
        with open('./exp/'+file2+'.p', 'rb') as f:
            try:
                while True:
                    data.append(pkl.load(f))
            except EOFError:
                pass
            
        print("data after second file:", len(data))
        
        learnerName = data[1]['alearners'][0]
        data[0]['alearners'].append(data[1]['alearners'][0])
        data[0]['performances'][learnerName] = data[1]['performances'][learnerName]
        
       
        with open('./exp/'+dest+'.p', 'wb') as fp:
            pkl.dump(data[0],fp)  
    
    def nameBC(self, alearner):
        if alearner == 'block-certainty-100' or alearner == 'block-certainty-k100':
            return 'block-certainty (K=100)'
        elif alearner == 'block-certainty-10' or alearner == 'block-certainty-k10':
            return 'block-certainty (K=10)'
        elif alearner == 'block-certainty-1' or alearner == 'block-certainty-k1':
            return 'block-certainty (K=1)'
        elif alearner == 'block-certainty-01' or alearner == 'block-certainty-k01':
            return 'block-certainty (K=0.1)'
        else:
            return 'block-certainty (K=0.01)'
    
    def valBC(self, val):
        if val == '100' or val == 'k100':
            return 100
        elif val == '10' or val == 'k10':
            return 10
        elif val == '1' or val == 'k1':
            return 1
        elif val == '01' or val == 'k01':
            return 0.1
        else:
            return 0.01
        
    def plotSingleResults(self, metrics = None):
        '''Plot the performance in the metrics, if metrics is not specified, plot all the metrics that were saved'''
        # add small epsilon to the denominator to avoid division by zero
        small_eps = 0.000001
        col = self._get_cmap(len(self.alearners)+1)
        
        if metrics is None:
            for performanceMeasure in self.performanceMeasures:
                plt.figure()
                i = 0
                for alearner in self.alearners:
                    avResult = np.mean(self.performances[alearner][performanceMeasure], axis=0)
                    plt.plot(avResult, color=col(i), label=alearner)
                    i = i+1.1
                plt.xlabel('Iteration count')
                plt.ylabel(performanceMeasure)
                plt.legend(loc='lower right', prop={'size': 6})
        else:
            for performanceMeasure in metrics:
                if performanceMeasure in self.existingMetrics:
                    plt.figure()
                    i = 0
                    for alearner in self.alearners:
                        if alearner.startswith('block'):
                            lName = self.nameBC(alearner)
                        else:
                            lName = alearner
                        if performanceMeasure=='accuracy':
                            avResult1 = np.mean(self.performances[alearner]['accuracy1'], axis=0)
                        elif performanceMeasure=='auc':
                            avResult1 =np.mean(self.performances[alearner]['auc1'], axis=(0))
                        elif performanceMeasure=='f1':
                            avResult1 =np.mean(self.performances[alearner]['fbeta11'], axis=(0)).tolist()
                            avResult1 = np.array([i[1] for i in avResult1])
                        elif performanceMeasure=='f3':
                            avResult1 =np.mean(self.performances[alearner]['fbeta31'], axis=(0)).tolist()
                            avResult1 = np.array([i[1] for i in avResult1])
                        elif performanceMeasure=='IoU':
                            avResult1 =np.mean((self.performances[alearner]['TP1']/(self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                        elif performanceMeasure=='dice':
                            avResult1 = np.mean((2*self.performances[alearner]['TP1']/(2*self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                        elif performanceMeasure=='f-measure':
                            avResult1 = np.mean((2*self.performances[alearner]['TP1']/(2*self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                        plt.plot(avResult1, color=col(i), label=lName + ' - only ML')
                        i = i+1.1
                    plt.xlabel('Iteration count')
                    plt.ylabel(performanceMeasure)
                    plt.legend(loc='lower right', prop={'size': 6})
                else:
                    print('This metric is not implemented, existing metrics = ', self.existingMetrics)
                    
        K = [0.01, 0.1, 1, 10, 100]
        
        for k in K:
            plt.figure()
            i = 0
            for alearner in self.alearners:
                if alearner.startswith('block'):
                    val = self.valBC(alearner.split('-')[2])
                    if val==k:
                        lName = self.nameBC(alearner)
                        cost = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                        plt.plot(cost, color=col(i), label=lName)
                        i += 1.5
                else:
                    lName = alearner
                    cost = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                    plt.plot(cost, color=col(i), label=lName)
                    i += 1.5
                    
            plt.xlabel('Iteration count')
            plt.ylabel("Cost (k={})".format(k))
            plt.legend(loc='lower right', prop={'size': 6})
            plt.show()
        
        plt.figure()
        i = 0
        for alearner in self.alearners:
            if alearner.startswith('block'):
                lName = self.nameBC(alearner)
            else:
                lName = alearner
            plt.plot(np.mean(self.performances[alearner]['numOfTrainedItems'], axis=0), color=col(i), label=lName)
            i += 1
        plt.xlabel('Iteration count')
        plt.ylabel('Number of classified items')
        plt.legend(loc='lower right', prop={'size': 6})
        plt.show()
    
    def plotResults(self, name, metrics = None):
        '''Plot the performance in the metrics, if metrics is not specified, plot all the metrics that were saved'''
        # add small epsilon to the denominator to avoid division by zero
        small_eps = 0.000001
        col = self._get_cmap(len(self.alearners)+1)
        
        if metrics is None:
            for performanceMeasure in self.performanceMeasures:
                plt.figure()
                i = 0
                for alearner in self.alearners:
                    avResult = np.mean(self.performances[alearner][performanceMeasure], axis=0)
                    plt.plot(avResult, color=col(i), label=alearner)
                    i = i+1.1
                plt.xlabel('Iteration count')
                plt.ylabel(performanceMeasure)
                plt.legend(loc='lower right', prop={'size': 6})
                plt.title(name)
        else:
            for performanceMeasure in metrics:
                if performanceMeasure in self.existingMetrics:
                    plt.figure()
                    i = 0
                    for alearner in self.alearners:
                        if alearner.startswith('block'):
                            lName = self.nameBC(alearner)
                        else:
                            lName = alearner
                        if performanceMeasure=='accuracy':
                            avResult1 = np.mean(self.performances[alearner]['accuracy1'], axis=0)
                            avResult2 = np.mean(self.performances[alearner]['accuracy2'], axis=0)
                        elif performanceMeasure=='auc':
                            avResult1 =np.mean(self.performances[alearner]['auc1'], axis=(0))
                            avResult2 =np.mean(self.performances[alearner]['auc2'], axis=(0))
                        elif performanceMeasure=='f1':
                            avResult1 =np.mean(self.performances[alearner]['fbeta11'], axis=(0)).tolist()
                            avResult1 = np.array([i[1] for i in avResult1])
                            avResult2 =np.mean(self.performances[alearner]['fbeta12'], axis=(0)).tolist()
                            avResult2 = np.array([i[1] for i in avResult2])
                        elif performanceMeasure=='f3':
                            avResult1 =np.mean(self.performances[alearner]['fbeta31'], axis=(0)).tolist()
                            avResult1 = np.array([i[1] for i in avResult1])
                            avResult2 =np.mean(self.performances[alearner]['fbeta32'], axis=(0)).tolist()
                            avResult2 = np.array([i[1] for i in avResult2])
                        elif performanceMeasure=='IoU':
                            avResult1 =np.mean((self.performances[alearner]['TP1']/(self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                            avResult2 =np.mean((self.performances[alearner]['TP2']/(self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                        elif performanceMeasure=='dice':
                            avResult1 = np.mean((2*self.performances[alearner]['TP1']/(2*self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                            avResult2 = np.mean((2*self.performances[alearner]['TP2']/(2*self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                        elif performanceMeasure=='f-measure':
                            avResult1 = np.mean((2*self.performances[alearner]['TP1']/(2*self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                            avResult2 = np.mean((2*self.performances[alearner]['TP2']/(2*self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                        plt.plot(avResult1, '--',  color=col(i), label=lName + ' - only ML')
                        plt.plot(avResult2, color=col(i), label=lName + ' - ML+Crowd')
                        i = i+1.1
                    plt.xlabel('Iteration count')
                    plt.ylabel(performanceMeasure)
                    plt.legend(loc='lower right', prop={'size': 6})
                    plt.title(name)
                else:
                    print('This metric is not implemented, existing metrics = ', self.existingMetrics)
                    
        K = [0.01, 0.1, 1, 10, 100]
        
        for k in K:
            plt.figure()
            i = 0
            for alearner in self.alearners:
                if alearner.startswith('block'):
                    val = self.valBC(alearner.split('-')[2])
                    if val==k:
                        lName = self.nameBC(alearner)
                        cost = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                        plt.plot(cost, color=col(i), label=lName)
                        i += 1.5
                else:
                    lName = alearner
                    cost = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                    plt.plot(cost, color=col(i), label=lName)
                    i += 1.5
                    
            plt.xlabel('Iteration count')
            plt.ylabel("Cost (k={})".format(k))
            plt.legend(loc='lower right', prop={'size': 6})
            plt.title(name)
            plt.show()
        
        plt.figure()
        i = 0
        for alearner in self.alearners:
            if alearner.startswith('block'):
                lName = self.nameBC(alearner)
            else:
                lName = alearner
            plt.plot(np.mean(self.performances[alearner]['numOfTrainedItems'], axis=0), color=col(i), label=lName)
            i += 1
        plt.xlabel('Iteration count')
        plt.ylabel('Number of classified items')
        plt.legend(loc='lower right', prop={'size': 6})
        plt.title(name)
        plt.show()
        
    def plotAggregationResults(self, resultFiles, figureName, aggTech, metrics = None):
        '''Plot the performance in the metrics, if metrics is not specified, plot all the metrics that were saved'''
        
        # add small epsilon to the denominator to avoid division by zero
        small_eps = 0.000001
        
        
        if metrics is None:
            for performanceMeasure in self.performanceMeasures:
                fig, axs = plt.subplots(2, 2, figsize=(20, 14))
                fig.suptitle(figureName)
                plots = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
                for ind in range(4):
                    self.readResult(resultFiles[ind])
                    col = self._get_cmap(len(self.alearners)+1)
                    i = 0
                    for alearner in self.alearners:
                        avResult = np.mean(self.performances[alearner][performanceMeasure], axis=0)
                        plots[ind].plot(avResult, color=col(i), label=alearner)
                        i = i+1.1
                    plots[ind].set_xlabel('Iteration count')
                    plots[ind].set_ylabel(performanceMeasure)
                    plots[ind].set_title(aggTech[ind])
                    plots[ind].legend(loc='upper left', prop={'size': 6})
                
        else:
            for performanceMeasure in metrics:
                fig, axs = plt.subplots(2, 2, figsize=(20, 14))
                fig.suptitle(figureName)
                plots = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
                for ind in range(4):
                    self.readResult(resultFiles[ind])
                    col = self._get_cmap(len(self.alearners)+1)
                    if performanceMeasure in self.existingMetrics:
                        i = 0
                        for alearner in self.alearners:
                            if alearner.startswith('block'):
                                lName = self.nameBC(alearner)
                            else:
                                lName = alearner
                            if performanceMeasure=='accuracy':
                                avResult1 = np.mean(self.performances[alearner]['accuracy1'], axis=0)
                                avResult2 = np.mean(self.performances[alearner]['accuracy2'], axis=0)
                            elif performanceMeasure=='auc':
                                avResult1 =np.mean(self.performances[alearner]['auc1'], axis=(0))
                                avResult2 =np.mean(self.performances[alearner]['auc2'], axis=(0))
                            elif performanceMeasure=='f1':
                                avResult1 =np.mean(self.performances[alearner]['fbeta11'], axis=(0)).tolist()
                                avResult1 = np.array([i[1] for i in avResult1])
                                avResult2 =np.mean(self.performances[alearner]['fbeta12'], axis=(0)).tolist()
                                avResult2 = np.array([i[1] for i in avResult2])
                            elif performanceMeasure=='f3':
                                avResult1 =np.mean(self.performances[alearner]['fbeta31'], axis=(0)).tolist()
                                avResult1 = np.array([i[1] for i in avResult1])
                                avResult2 =np.mean(self.performances[alearner]['fbeta32'], axis=(0)).tolist()
                                avResult2 = np.array([i[1] for i in avResult2])
                            elif performanceMeasure=='IoU':
                                avResult1 =np.mean((self.performances[alearner]['TP1']/(self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                                avResult2 =np.mean((self.performances[alearner]['TP2']/(self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                            elif performanceMeasure=='dice':
                                avResult1 = np.mean((2*self.performances[alearner]['TP1']/(2*self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                                avResult2 = np.mean((2*self.performances[alearner]['TP2']/(2*self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                            elif performanceMeasure=='f-measure':
                                avResult1 = np.mean((2*self.performances[alearner]['TP1']/(2*self.performances[alearner]['TP1']+self.performances[alearner]['FP1']+self.performances[alearner]['FN1']+small_eps)),axis=(0))
                                avResult2 = np.mean((2*self.performances[alearner]['TP2']/(2*self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                            plots[ind].plot(avResult1, '--',  color=col(i), label=lName + ' - only ML')
                            plots[ind].plot(avResult2, color=col(i), label=lName + ' - ML+Crowd')
                            i = i+1.1
                        plots[ind].set_xlabel('Iteration count')
                        plots[ind].set_ylabel(performanceMeasure)
                        plots[ind].set_title(aggTech[ind])
                        plots[ind].legend(loc='upper left', prop={'size': 6})
                    else:
                        print('This metric is not implemented, existing metrics = ', self.existingMetrics)
                    
        K = [0.01, 0.1, 1, 10, 100]
        for k in K:
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            fig.suptitle(figureName)
            plots = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
            for ind in range(4):
                self.readResult(resultFiles[ind])
                col = self._get_cmap(len(self.alearners)+1)
                i = 0
                for alearner in self.alearners:
                    if alearner.startswith('block'):
                        val = self.valBC(alearner.split('-')[2])
                        if val==k:
                            lName = self.nameBC(alearner)
                            cost = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                            plots[ind].plot(cost, color=col(i), label=lName)
                            i += 1.5
                    else:
                        lName = alearner
                        cost = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                        plots[ind].plot(cost, color=col(i), label=lName)
                        i += 1.5
                    
                plots[ind].set_xlabel('Iteration count')
                plots[ind].set_ylabel("Cost (k={})".format(k))
                plots[ind].set_title(aggTech[ind])
                plots[ind].legend(loc='upper right', prop={'size': 7})

        
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle(figureName)
        plots = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
        for ind in range(4):
            self.readResult(resultFiles[ind])
            col = self._get_cmap(len(self.alearners)+1)
            i = 0
            for alearner in self.alearners:
                if alearner.startswith('block'):
                    lName = self.nameBC(alearner)
                else:
                    lName = alearner
                plots[ind].plot(np.mean(self.performances[alearner]['numOfTrainedItems'], axis=0), color=col(i), label=lName)
                i += 1
            plots[ind].set_xlabel('Iteration count')
            plots[ind].set_ylabel('Number of classified items')
            plots[ind].set_title(aggTech[ind])
            plots[ind].legend(loc='lower right', prop={'size': 7})
            
            
    def plotAggregationClfResults(self, resultFiles, figureName, aggTech, metrics = None):
        '''Plot the performance in the metrics, if metrics is not specified, plot all the metrics that were saved'''
        
        # add small epsilon to the denominator to avoid division by zero
        small_eps = 0.000001
        
        
        if metrics is None:
            for performanceMeasure in self.performanceMeasures:
                fig, axs = plt.subplots(2, 2, figsize=(20, 14))
                fig.suptitle(figureName)
                plots = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
                for ind in range(4):
                    for inn in range(2):
                        self.readResult(resultFiles[ind][inn])
                        col = self._get_cmap(len(self.alearners)+1)
                        i = 0
                        for alearner in self.alearners:
                            avResult = np.mean(self.performances[alearner][performanceMeasure], axis=0)
                            if inn == 0:
                                plots[ind].plot(avResult,'--',  color=col(i), label=alearner + ' - ML+Crowd - RF')
                            elif inn == 1:
                                plots[ind].plot(avResult, color=col(i), label=alearner + ' - ML+Crowd - SVM')
                            i = i+1.1
                        plots[ind].set_xlabel('Iteration count')
                        plots[ind].set_ylabel(performanceMeasure)
                        plots[ind].set_title(aggTech[ind])
                        plots[ind].legend(loc='upper left', prop={'size': 6})
                
        else:
            for performanceMeasure in metrics:
                fig, axs = plt.subplots(2, 2, figsize=(20, 14))
                fig.suptitle(figureName)
                plots = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
                for ind in range(4):
                    for inn in range(2):
                        self.readResult(resultFiles[ind][inn])
                        col = self._get_cmap(len(self.alearners)+1)
                        if performanceMeasure in self.existingMetrics:
                            i = 0
                            for alearner in self.alearners:
                                proceed = 1
                                if alearner.startswith('block'):
                                    lName = self.nameBC(alearner)
                                    if lName in ['block-certainty (K=100)', 'block-certainty (K=10)', 'block-certainty (K=0.1)', 'block-certainty (K=0.01)']:
                                        proceed = 0
                                else:
                                    lName = alearner
                                if proceed == 1:
                                    if performanceMeasure=='accuracy':
                                        avResult = np.mean(self.performances[alearner]['accuracy2'], axis=0)
                                    elif performanceMeasure=='auc':
                                        avResult =np.mean(self.performances[alearner]['auc2'], axis=(0))
                                    elif performanceMeasure=='f1':
                                        avResult =np.mean(self.performances[alearner]['fbeta12'], axis=(0)).tolist()
                                        avResult = np.array([i[1] for i in avResult])
                                    elif performanceMeasure=='f3':
                                        avResult =np.mean(self.performances[alearner]['fbeta32'], axis=(0)).tolist()
                                        avResult = np.array([i[1] for i in avResult])
                                    elif performanceMeasure=='IoU':
                                        avResult =np.mean((self.performances[alearner]['TP2']/(self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                                    elif performanceMeasure=='dice':
                                        avResult = np.mean((2*self.performances[alearner]['TP2']/(2*self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                                    elif performanceMeasure=='f-measure':
                                        avResult = np.mean((2*self.performances[alearner]['TP2']/(2*self.performances[alearner]['TP2']+self.performances[alearner]['FP2']+self.performances[alearner]['FN2']+small_eps)),axis=(0))
                                    if inn == 0:
                                        plots[ind].plot(avResult, '--', color=col(i), label=lName + ' - ML+Crowd - RF')
                                    elif inn == 1:
                                        plots[ind].plot(avResult, color=col(i), label=lName + ' - ML+Crowd - SVM')
                                    i = i+1.1
                            plots[ind].set_xlabel('Iteration count')
                            plots[ind].set_ylabel(performanceMeasure)
                            plots[ind].set_title(aggTech[ind])
                            plots[ind].legend(loc='upper left', prop={'size': 7})
                        else:
                            print('This metric is not implemented, existing metrics = ', self.existingMetrics)
                    
        K = [0.01, 0.1, 1, 10, 100]
        for k in K:
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            fig.suptitle(figureName)
            plots = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
            for ind in range(4):
                for inn in range(2):
                    self.readResult(resultFiles[ind][inn])
                    col = self._get_cmap(len(self.alearners)+1)
                    i = 0
                    for alearner in self.alearners:
                        if alearner.startswith('block'):
                            val = self.valBC(alearner.split('-')[2])
                            if val==k:
                                lName = self.nameBC(alearner)
                                cost = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                                if inn == 0:
                                    plots[ind].plot(cost, '--', color=col(i), label=lName + ' - ML+Crowd - RF')
                                elif inn == 1:
                                    plots[ind].plot(cost, color=col(i), label=lName + ' - ML+Crowd - SVM')
                                i += 1.5
                        else:
                            lName = alearner
                            cost = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                            if inn == 0:
                                plots[ind].plot(cost, '--', color=col(i), label=lName + ' - ML+Crowd - RF')
                            elif inn == 1:
                                plots[ind].plot(cost, color=col(i), label=lName + ' - ML+Crowd - SVM')
                            i += 1.5
                        
                    plots[ind].set_xlabel('Iteration count')
                    plots[ind].set_ylabel("Cost (k={})".format(k))
                    plots[ind].set_title(aggTech[ind])
                    plots[ind].legend(loc='upper right', prop={'size': 7})

        
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle(figureName)
        plots = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
        for ind in range(4):
            for inn in range(2):
                self.readResult(resultFiles[ind][inn])
                col = self._get_cmap(len(self.alearners)+1)
                i = 0
                for alearner in self.alearners:
                    if alearner.startswith('block'):
                        lName = self.nameBC(alearner)
                    else:
                        lName = alearner
                    if inn == 0:
                        plots[ind].plot(np.mean(self.performances[alearner]['numOfTrainedItems'], axis=0), '--', color=col(i), label=lName+ ' - ML+Crowd - RF')
                    elif inn == 1:
                        plots[ind].plot(np.mean(self.performances[alearner]['numOfTrainedItems'], axis=0), color=col(i), label=lName+ ' - ML+Crowd - SVM')
                    i += 1
                plots[ind].set_xlabel('Iteration count')
                plots[ind].set_ylabel('Number of classified items')
                plots[ind].set_title(aggTech[ind])
                plots[ind].legend(loc='lower right', prop={'size': 7})
    
    def printResults(self):
        
        result = {}
        for alearner in self.alearners:
            result[alearner] = {}
            acc1 = np.mean(self.performances[alearner]['accuracy1'], axis=0)[-1]
            result[alearner]['accuracy1'] = acc1
            acc2 = np.mean(self.performances[alearner]['accuracy2'], axis=0)[-1]
            result[alearner]['accuracy2'] = acc2
            fbeta11 = np.mean(self.performances[alearner]['fbeta11'], axis=0)[-1][1]
            result[alearner]['fbeta11'] = fbeta11
            fbeta12 = np.mean(self.performances[alearner]['fbeta12'], axis=0)[-1][1]
            result[alearner]['fbeta12'] = fbeta12
            fbeta31 = np.mean(self.performances[alearner]['fbeta31'], axis=0)[-1][1]
            result[alearner]['fbeta31'] = fbeta31
            fbeta32 = np.mean(self.performances[alearner]['fbeta32'], axis=0)[-1][1]
            result[alearner]['fbeta32'] = fbeta32   
                
        K = [100, 10, 1, 0.1, 0.01]
        
        for k in K:
            for alearner in self.alearners:
                if alearner.startswith('block'):
                    val = self.valBC(alearner.split('-')[2])
                    if val==k:
                        cost1 = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                        cost2 = (k*np.mean(self.performances[alearner]['FN2'], axis=0)+np.mean(self.performances[alearner]['FP2'], axis=0))/len(self.performances[alearner]['FP1'])
                        result[alearner]['cost1_{}'.format(k)] = cost1[-1]
                        result[alearner]['cost2_{}'.format(k)] = cost2[-1]
                else:
                    cost1 = (k*np.mean(self.performances[alearner]['FN1'], axis=0)+np.mean(self.performances[alearner]['FP1'], axis=0))/len(self.performances[alearner]['FP1'])
                    cost2 = (k*np.mean(self.performances[alearner]['FN2'], axis=0)+np.mean(self.performances[alearner]['FP2'], axis=0))/len(self.performances[alearner]['FP1'])
                    result[alearner]['cost1_{}'.format(k)] = cost1[-1]
                    result[alearner]['cost2_{}'.format(k)] = cost2[-1]
        
        for alearner in self.alearners:
            numOfTrainedItems = np.mean(self.performances[alearner]['numOfTrainedItems'], axis=0)
            result[alearner]['numOfTrainedItems'] = numOfTrainedItems[-1]
            
        return result
    
    def _get_cmap(self, N):
        '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
        RGB color.'''
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)
        return map_index_to_rgb_color