#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:30:42 2020

@author: burcusyn
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.append('./Classes/')
# import various AL strategies
from active_learner import ActiveLearnerRandom
from active_learner import ActiveLearnerUncertainty
from active_learner import ActiveLearnerCertainty
from active_learner import ActiveLearnerBlockCertainty
from active_learner import ActiveLearnerQUIRE
from active_learner import ActiveLearnerLAL
from active_learner import ActiveLearnerMinExpError

# import the dataset class
from dataset import DatasetRecognizingTextualEntailment
from dataset import Emotion
from dataset import AmazonSentiment
from dataset import Crisis
from dataset import Exergame

# import Experiment and Result classes that will be responsible for running AL and saving the results
from experiment import Experiment
from results import Results


if __name__ == '__main__':                  
    
    # number of experiment repeats
    nExperiments = 20

    # number of labeled points at the beginning of the AL experiment
    nStart = 20
    # number of iterations in AL experiment
    nIterations = 90
    # number of items to be queried at each iteration
    batchSize = 20
    # maximum number of votes to ask per item np.inf or 3
    maxVoteCount = np.inf
    # vote aggregation technique can be either majorityVoting (majorityVoting) or Dawid-Skene (DS)
    voteAggTechnique = 'majorityVoting'

    # the quality metrics computed on the test set to evaluate active learners
    quality_metrics = ['numOfTrainedItems', 'accuracy1', 'accuracy2', 'TN1', 'TN2', 'TP1', 'TP2', 'FN1', 'FN2', 'FP1', 'FP2', 'fbeta11', 'fbeta12', 'fbeta31', 'fbeta32','auc1', 'auc2']
    
    # load dataset
    dtst = Crisis('author_cleaned')
    
    
    # set the starting point
    dtst.setStartState(nStart)
    
    #Set classifier model
    #if you want to use SVM Classifier, uncomment line 70, and comment the line 69, note that LAL models will again work with Radnom Forest Classifier
    model = RandomForestClassifier(class_weight='balanced', random_state=2020)  
#    model = CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=0.1))
    
    # Build classifiers for LAL strategies
    fn = 'LAL-randomtree-simulatedunbalanced-big.npz'
    # Konyushkova et al. found these parameters by cross-validating the regressor and now we reuse these expreiments
    parameters = {'est': 2000, 'depth': 40, 'feat': 6 }
    filename = './lal datasets/'+fn
    regression_data = np.load(filename)
    regression_features = regression_data['arr_0']
    regression_labels = regression_data['arr_1']
    
    print('Building lal regression model..')
    lalModel1 = RandomForestRegressor(n_estimators = parameters['est'], max_depth = parameters['depth'], 
                                     max_features=parameters['feat'], oob_score=True, n_jobs=8)
    
    lalModel1.fit(regression_features, np.ravel(regression_labels))    
    
    n = 'LAL-iterativetree-simulatedunbalanced-big.npz'
    # Konyushkova et al. found these parameters by cross-validating the regressor and now we reuse these expreiments
    parameters = {'est': 1000, 'depth': 40, 'feat': 6 }
    filename = './lal datasets/'+fn
    regression_data = np.load(filename)
    regression_features = regression_data['arr_0']
    regression_labels = regression_data['arr_1']
    
    print('Building lal regression model..')
    lalModel2 = RandomForestRegressor(n_estimators = parameters['est'], max_depth = parameters['depth'], 
                                     max_features=parameters['feat'], oob_score=True, n_jobs=8)
    
    lalModel2.fit(regression_features, np.ravel(regression_labels)) 
    
#    # Active learning strategies
    alR = ActiveLearnerRandom(dtst, 'random', model, batchSize, voteAggTechnique)
    alU = ActiveLearnerUncertainty(dtst, 'uncertainty', model, batchSize, voteAggTechnique)
    alC = ActiveLearnerCertainty(dtst, 'certainty', model, batchSize, voteAggTechnique)
    alBC100 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-k100', model, batchSize, 100, voteAggTechnique)
    alBC10 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-k10', model, batchSize, 10, voteAggTechnique)
    alBC1 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-k1', model, batchSize, 1, voteAggTechnique)
    alBC01 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-k01', model, batchSize, 0.1, voteAggTechnique)
    alBC001 = ActiveLearnerBlockCertainty(dtst, 'block-certainty-k001', model, batchSize, 0.01, voteAggTechnique)
    alQ = ActiveLearnerQUIRE(dtst, 'quire', model, batchSize, voteAggTechnique, 1.0, 1., 'rbf', 1., 3)
    als = [alR, alU, alC, alBC100, alBC10, alBC1, alBC01, alBC001, alQ]
    alLALindepend = ActiveLearnerLAL(dtst, 'lal-rand', model, batchSize, lalModel1, voteAggTechnique)
    alLALiterative = ActiveLearnerLAL(dtst, 'lal-iter', model, batchSize, lalModel2, voteAggTechnique)
    alMinExp = ActiveLearnerMinExpError(dtst, 'minExpError', model, batchSize, voteAggTechnique)

    als = [alR, alU, alC, alBC100, alBC10, alBC1, alBC01, alBC001, alQ, alLALindepend, alLALiterative, alMinExp]
    
   
    exp = Experiment(nIterations, quality_metrics, dtst, als, maxVoteCount, 'here we can put a comment about the current experiments')
   # the Results class helps to add, save and plot results of the experiments
    res = Results(exp, nExperiments)
    
    print("Running AL experiments...")
    for i in range(nExperiments):
        print('\n experiment #'+str(i+1))
        # run an experiment
        performance = exp.run()
        res.addPerformance(performance)
        # reset the experiment (including sampling a new starting state for the dataset)
        exp.reset()
    
    print("Done!")
    res.saveResults('crisis_author_inf_voteAggMV_svm')
    