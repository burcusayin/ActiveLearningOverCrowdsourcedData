#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:31:56 2019

@author: burcusyn
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import random
import numpy as np
import sys

class Vectorizer():
    ''' The class that transforms text data to vectors. '''
    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=3, max_features=None, 
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
            stop_words = None, lowercase=False)

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

    def fit(self, X):
        self.vectorizer.fit(X)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X).toarray()
    
class VoteAggretagor():
    ''' The class that includes methods for vote aggregation. '''
    def majorityVoting(self, voteHistory, datapoint):
        votes = voteHistory[datapoint]
        vote_count = Counter(votes)
        vote_count_values = list(vote_count.values())
        if len(vote_count_values) > 1 and (vote_count_values[0] == vote_count_values[1]):
            majorityVote = random.sample([0,1], 1)[0]
        else:
            majorityVote = int(vote_count.most_common(1)[0][0])
        return majorityVote
    
    # D&S vote aggregation method is adapted to the code written by Sukrut Rao in Fast-Dawid-Skene method
    # github link to the code: https://github.com/sukrutrao/Fast-Dawid-Skene
    def DS(self, voteHistoryAll, workerHistoryAll, datapoint):
        voteHistory = list(voteHistoryAll[datapoint])
        participants = list(workerHistoryAll[datapoint])
        questions = list([datapoint])
        
        nQuestions = 1
        nParticipants = len(participants)
        classes = list([0, 1])
        nClasses = len(classes)
        
        # create a 3d array to hold counts
        counts = np.zeros([nQuestions, nParticipants, nClasses])
        
        # convert responses to counts
        for question in questions:
            i = questions.index(question)
            if len(participants) != 0:
                for participant in participants:
                    k = participants.index(participant)
                    j = classes.index(voteHistory[k])
                    counts[i, k, j] += 1
        
        [nQuestions, nParticipants, nClasses] = np.shape(counts)
        response_sums = np.sum(counts, 1)
        question_classes = np.zeros([nQuestions, nClasses])
        for p in range(nQuestions):
            question_classes[p, :] = response_sums[p, :] / np.sum(response_sums[p, :], dtype=float)
        
        # initialize the DS 
        nIter = 0
        converged = False
        old_class_marginals = None
        tol=0.0001
        max_iter=100
        # total_time = 0
    
        while not converged:
            nIter += 1
    
            # Start measuring time
            # start = time.time()
    
            # M-step
            (class_marginals, error_rates) = EM.m_step(counts, question_classes)
    
            # E-step
            question_classes = EM.e_step(counts, class_marginals, error_rates)
    
            # End measuring time
            # end = time.time()
            # total_time += end-start
    
            # check likelihood
            log_L = EM.calc_likelihood(counts, class_marginals, error_rates)
    
            # check for convergence
            if old_class_marginals is not None:
                class_marginals_diff = np.sum(
                    np.abs(class_marginals - old_class_marginals))
        #        error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
                if (class_marginals_diff < tol) or nIter >= max_iter:
                    converged = True
    
            old_class_marginals = class_marginals
       #     old_error_rates = error_rates
    
        np.set_printoptions(precision=2, suppress=True)
    
        result = np.argmax(question_classes, axis=1)
        return result

# code for D&S vote aggregation method, written by Sukrut Rao
# github link to the code: https://github.com/sukrutrao/Fast-Dawid-Skene
class EM:
    def m_step(counts, question_classes):
        """
        M Step for the EM algorithm
        Get estimates for the prior class probabilities (p_j) and the error
        rates (pi_jkl) using MLE with current estimates of true question classes
        See equations 2.3 and 2.4 in Dawid-Skene (1979) or equations 3 and 4 in 
        our paper (Fast Dawid-Skene: A Fast Vote Aggregation Scheme for Sentiment 
        Classification)
        Args: 
            counts: Array of how many times each response was received
                by each question from each participant: [questions x participants x classes]
            question_classes: Matrix of current assignments of questions to classes
        Returns:
            p_j: class marginals - the probability that the correct answer of a question
                is a given choice (class) [classes]
            pi_kjl: error rates - the probability of participant k labeling
                response l for a question whose correct answer is j [participants, classes, classes]
        """
    
        [nQuestions, nParticipants, nClasses] = np.shape(counts)
    
        # compute class marginals
        class_marginals = np.sum(question_classes, 0) / float(nQuestions)
    
        # compute error rates
        error_rates = np.zeros([nParticipants, nClasses, nClasses])
        for k in range(nParticipants):
            for j in range(nClasses):
                for l in range(nClasses):
                    error_rates[k, j, l] = np.dot(
                        question_classes[:, j], counts[:, k, l])
                sum_over_responses = np.sum(error_rates[k, j, :])
                if sum_over_responses > 0:
                    error_rates[k, j, :] = error_rates[
                        k, j, :] / float(sum_over_responses)
    
        return (class_marginals, error_rates)


    def e_step(counts, class_marginals, error_rates):
        """
        E (+ C) Step for the EM algorithm
        Determine the probability of each question belonging to each class,
        given current ML estimates of the parameters from the M-step. Also 
        perform the C step (along with E step (see section 3.4)) in case of FDS.
        See equation 2.5 in Dawid-Skene (1979) or equations 1 and 2 in 
        our paper (Fast Dawid Skene: A Fast Vote Aggregation Scheme for Sentiment 
        Classification)
        Args:
            counts: Array of how many times each response was received
                by each question from each participant: [questions x participants x classes]
            class_marginals: probability of a random question belonging to each class: [classes]
            error_rates: probability of participant k assigning a question whose correct 
                label is j the label l: [participants x classes x classes]
        Returns:
            question_classes: Assignments of labels to questions
                [questions x classes]
        """
    
        [nQuestions, nParticipants, nClasses] = np.shape(counts)
    
        question_classes = np.zeros([nQuestions, nClasses])
    
        for i in range(nQuestions):
            for j in range(nClasses):
                estimate = class_marginals[j]
                estimate *= np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))
    
                question_classes[i, j] = estimate

            question_sum = np.sum(question_classes[i, :])
            if question_sum > 0:
                question_classes[i, :] = question_classes[
                    i, :] / float(question_sum)
    
        return question_classes


    def calc_likelihood(counts, class_marginals, error_rates):
        """
        Calculate the likelihood with the current  parameters
        Calculate the likelihood given the current parameter estimates
        This should go up monotonically as EM proceeds
        See equation 2.7 in Dawid-Skene (1979)
        Args:
            counts: Array of how many times each response was received
                by each question from each participant: [questions x participants x classes]
            class_marginals: probability of a random question belonging to each class: [classes]
            error_rates: probability of participant k assigning a question whose correct 
                label is j the label l: [observers x classes x classes]
        Returns:
            Likelihood given current parameter estimates
        """
    
        [nPatients, nObservers, nClasses] = np.shape(counts)
        log_L = 0.0
    
        for i in range(nPatients):
            patient_likelihood = 0.0
            for j in range(nClasses):
    
                class_prior = class_marginals[j]
                patient_class_likelihood = np.prod(
                    np.power(error_rates[:, j, :], counts[i, :, :]))
                patient_class_posterior = class_prior * patient_class_likelihood
                patient_likelihood += patient_class_posterior
    
            temp = log_L + np.log(patient_likelihood)
    
            if np.isnan(temp) or np.isinf(temp):
                print(i, log_L, np.log(patient_likelihood), temp)
                sys.exit()
    
            log_L = temp
    
        return log_L