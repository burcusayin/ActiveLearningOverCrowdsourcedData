#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:22:35 2019

@author: burcusyn
"""

class Experiment:
    '''The class that runs active learning experiment'''
    
    def __init__(self, nIterations, performanceMeasures, dataset, alearners, maxVoteCount, comment=''):
        
        self.nIterations = nIterations
        self.performanceMeasures = performanceMeasures
        self.dataset = dataset
        self.alearners = alearners
        self.maxVoteCount = maxVoteCount
        self.comment = comment
        self.performances = dict()
        for alearner in self.alearners:
            self.performances[alearner.name] = dict()
            for performanceMeasure in self.performanceMeasures:
                self.performances[alearner.name][performanceMeasure] = []

        
    def run(self):
        '''Run the experiment for nIterations for all alearners and return performances'''
        for it in range(self.nIterations):
            print('.', end="")
            for alearner in self.alearners:
                alearner.train()
                perf = alearner.evaluate(self.performanceMeasures)
                for key in perf:
                    self.performances[alearner.name][key].append(perf[key])
                datapoints = alearner.selectNext()
                alearner.update(datapoints, self.maxVoteCount)
        return self.performances
    
    
    def reset(self):
        '''Reset the experiment: reset the starting datapoint of the dataset, reset alearners and performances'''
        self.dataset.setStartState(self.dataset.nStart)
        for alearner in self.alearners:
            alearner.reset()
            
        self.performances = dict()
        for alearner in self.alearners:
            self.performances[alearner.name] = dict()
            for performanceMeasure in self.performanceMeasures:
                self.performances[alearner.name][performanceMeasure] = []