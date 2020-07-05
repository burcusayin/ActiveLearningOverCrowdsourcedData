#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:21:57 2019

@author: burcusyn
"""

import numpy as np
import pandas as pd 
from utils import Vectorizer


class Dataset:
    
    def __init__(self):
        
        # each dataset will have a pool of data, its gold labels and worker responses. 
        self.poolData = np.array([])
        self.poolGoldLabels = np.array([])
        self.poolWorkerResponses = np.array([])
        self.poolWorkers = np.array([])
        
    def setStartState(self, nStart):
        ''' This functions initialises fields indicesTrained and indicesUnknown which contain the datapoints having final labels(known) and still explorable(unknown) ones.
        Input:
        nStart -- number of labelled datapoints (size of indicesTrained)
        '''
        self.nStart = nStart
        self.indicesKnown = np.array([])
        self.indicesUnknown = np.array([])
        
        # get predefined positive and negative points so that both classes are represented and initial classifier could be trained.
        pos, neg = nStart // 2, nStart // 2
        for i in range(len(self.poolGoldLabels)):
            if pos > 0 and self.poolGoldLabels[i] == 1:
                if self.indicesKnown.size == 0:
                    self.indicesKnown = np.array([i])
                else:
                    self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([i])])); 
                pos -= 1
            elif neg > 0 and self.poolGoldLabels[i] == 0:
                if self.indicesKnown.size == 0:
                    self.indicesKnown = np.array([i])
                else:
                    self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([i])]));
                neg -= 1
                
            if pos == 0 and neg == 0:
                break
        
        for i in range(len(self.poolData)):
            if i not in self.indicesKnown:
                if self.indicesUnknown.size == 0:
                    self.indicesUnknown = np.array([i])
                else:
                    self.indicesUnknown = np.concatenate(([self.indicesUnknown, np.array([i])]));
        
class DatasetRecognizingTextualEntailment(Dataset):
    '''Loads the Recognizing Textual Entailment dataset.
    Origin of the dataset: Snow et al. 2008 - "Cheap and Fast -- But is it Good?  
    Evaluating Non-Expert Annotations for Natural Language Tasks", as published in the Proceedings of EMNLP 2008.
    https://sites.google.comß/site/nlpannotations/ '''
    
    def __init__(self, predicate):
        Dataset.__init__(self)
                
        filename = ''
        
        if predicate == 'bert':
            filename = './data/rte_transformed_dataset_bert.csv'
        elif predicate == 'cleaned':
            filename = './data/rte_transformed_dataset_cleaned_stem.csv'
        
        dt = pd.read_csv(filename)
        
        ''' group the data points and merge worker responses for each data point '''
        dt = dt.groupby(['taskContent','taskID','goldLabel']).agg(workerID=('workerID',lambda x: list(x)), response=('response',lambda x: list(x))).reset_index()
        
        y = dt['goldLabel'].values
        r = dt['response'].values.tolist()
        X = dt['taskContent'].values
        w = dt['workerID'].values
        
        v = Vectorizer()
        v.fit(X.astype('U'))
        X = v.transform(X.astype('U'))
        
        self.poolData = X
        self.poolGoldLabels = y
        self.poolWorkerResponses = r
        self.poolWorkers = w
        
class Emotion(Dataset):
    ''' Loads the Emotion dataset.
    Origin of the dataset: Snow et al. 2008 - "Cheap and Fast -- But is it Good?  
    Evaluating Non-Expert Annotations for Natural Language Tasks", as published in the Proceedings of EMNLP 2008.
    https://sites.google.comß/site/nlpannotations/ '''
    
    def __init__(self, predicate):
        Dataset.__init__(self)
                
        filename = ''
        if predicate == 'bert':
            filename = './data/emotion_anger_transformed_dataset_bert.csv'
        elif predicate == 'cleaned':
            filename = './data/emotion_anger_transformed_dataset_cleaned_stem.csv'
            
        dt = pd.read_csv(filename)
        
        dt['response'] = dt['response'].apply(lambda x: 0 if x <= 49 else 1)
        dt['goldLabel'] = dt['goldLabel'].apply(lambda x: 0 if x <= 49 else 1)
        
        ''' group the data points and merge worker responses for each data point '''
        dt = dt.groupby(['taskContent','taskID','goldLabel']).agg(workerID=('workerID',lambda x: list(x)), response=('response',lambda x: list(x))).reset_index()
        
        y = dt['goldLabel'].values
        r = dt['response'].values.tolist()
        X = dt['taskContent'].values
        w = dt['workerID'].values
        
        v = Vectorizer()
        v.fit(X.astype('U'))
        X = v.transform(X.astype('U'))
        
        self.poolData = X
        self.poolGoldLabels = y
        self.poolWorkerResponses = r
        self.poolWorkers = w
        
class AmazonSentiment(Dataset):
    ''' Loads Amazon Sentiment Dataset.
    Origin of the dataset: Krivosheev et al. 
    https://github.com/Evgeneus/screening-classification-datasets/tree/master/crowdsourced-amazon-sentiment-dataset/data'''
    
    def __init__(self, predicate):
        Dataset.__init__(self)
        
        filename = ''
        if predicate == 'is_book_bert':
            filename = './data/1k_amazon_reviews_is_book_bert.csv'
        elif predicate == 'is_book_cleaned':
            filename = './data/1k_amazon_reviews_is_book_cleaned_stem.csv'
        elif predicate == 'is_negative_bert':
            filename = './data/1k_amazon_reviews_is_negative_bert.csv'
        elif predicate == 'is_negative_cleaned':
            filename = './data/1k_amazon_reviews_is_negative_cleaned_stem.csv'
            
        dt = pd.read_csv(filename)
        
        ''' group the data points and merge worker responses for each data point '''
        dt = dt.groupby(['taskContent','taskID','goldLabel']).agg(workerID=('workerID',lambda x: list(x)), response=('response',lambda x: list(x))).reset_index()
        
        y = dt['goldLabel'].values
        r = dt['response'].values.tolist()
        X = dt['taskContent'].values
        w = dt['workerID'].values
        
        v = Vectorizer()
        v.fit(X.astype('U'))
        X = v.transform(X.astype('U'))
        
        self.poolData = X
        self.poolGoldLabels = y
        self.poolWorkerResponses = r
        self.poolWorkers = w
        
class Crisis(Dataset):
    ''' Loads Crisis-1 Dataset.
    Origin of the dataset: Imran et al. 2013 - " Practical Extraction of Disaster-Relevant Information from Social Media", as 
    published in Proceedings of the 22nd international conference on World Wide Web companion, May 2013, Rio de Janeiro, Brazil.
    https://crisisnlp.qcri.org/ '''
    def __init__(self, predicate):
        Dataset.__init__(self)
        filename = ''
        if predicate == 'author_bert':
            filename = './data/crisis1_author_of_tweet_transformed_dataset_bert.csv'
        elif predicate == 'author_cleaned':
            filename = './data/crisis1_author_of_tweet_transformed_dataset_cleaned_stem.csv'
        elif predicate == 'message_bert':
            filename = './data/crisis1_type_of_message_transformed_dataset_bert.csv'
        elif predicate == 'message_cleaned':
            filename = './data/crisis1_type_of_message_transformed_dataset_cleaned_stem.csv'
        elif predicate == 'type_bert':
            filename = './data/crisis2_determine_type_of_hurricane_related_tweet_bert.csv'
        elif predicate == 'type_cleaned':
            filename = './data/crisis2_determine_type_of_hurricane_related_tweet_cleaned_stem.csv'
        dt = pd.read_csv(filename)

        ''' group the data points and merge worker responses for each data point '''
        dt = dt.groupby(['taskContent', 'taskID', 'goldLabel']).agg(workerID=('workerID', lambda x: list(x)), response=( 'response', lambda x: list(x))).reset_index()

        y = dt['goldLabel'].values
        r = dt['response'].values.tolist()
        X = dt['taskContent'].values
        w = dt['workerID'].values

        v = Vectorizer()
        v.fit(X.astype('U'))
        X = v.transform(X.astype('U'))

        self.poolData = X
        self.poolGoldLabels = y
        self.poolWorkerResponses = r
        self.poolWorkers = w
        
class Exergame(Dataset):
    ''' Loads Exergame Dataset'''
    def __init__(self):
        Dataset.__init__(self)
                
        filename = './data/exergame_dataset_cleaned_stem.csv'
        
        dt = pd.read_csv(filename)
        
        ''' group the data points and merge worker responses for each data point '''
        dt = dt.groupby(['taskContent','taskID','goldLabel']).agg(workerID=('workerID',lambda x: list(x)), response=('response',lambda x: list(x))).reset_index()
        
        y = dt['goldLabel'].values
        r = dt['response'].values.tolist()
        X = dt['taskContent'].values
        w = dt['workerID'].values
        
        v = Vectorizer()
        v.fit(X.astype('U'))
        X = v.transform(X.astype('U'))
        
        self.poolData = X
        self.poolGoldLabels = y
        self.poolWorkerResponses = r
        self.poolWorkers = w
        
    
        