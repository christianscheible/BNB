#!/usr/bin/env python

import numpy as np
import random

import cPickle
from sys import stderr, exit
from collections import defaultdict
from operator import itemgetter
from itertools import izip
from copy import deepcopy

import util
from corpus import Corpus



class Model:
  def __init__(self, corpus, n_labels=2, iterations=100, \
               gamma_pi=.01, gamma_theta=.01, debug=True,\
               lag=10, burn_in=10):
    
    # copy parameters
    self.corpus     = corpus
    self.iterations = iterations
    self.gamma_pi   = gamma_pi
    self.gamma_theta= gamma_theta
    self.n_labels   = n_labels
    self.V          = self.corpus.n_tokens
    self.N          = self.corpus.n_types
    self.lag        = lag
    self.burn_in    = burn_in
    
    self.debug      = debug
   
       
    # probabilities
    self.theta      = [None] * self.n_labels
    self.theta_hist = []
    self.avg_theta  = None
    
    
    # structure
    self.label_assign = [None] * self.corpus.n_docs
    
    # counts
    self.c_word_label = [None] * self.n_labels
    self.c_label      = [0] * self.n_labels
    
    # misc
    self.rng = random.Random(12345)
  
    self.initialize()
    
    if self.debug:
      print 'vocabulary:', self.corpus.vocabulary
  
  
  def print_parameters(self):
    stderr.write("iterations: %d\n" % self.iterations)
    stderr.write("burn-in: %d\n" % self.burn_in)
    stderr.write("lag: %d\n" % self.lag)
    stderr.write("gamma_pi: %f\n" % self.gamma_pi)
    stderr.write("gamma_theta: %f\n" % self.gamma_theta)
  
  

  def initialize(self, random_topics=False):
    # initialize label counts to zero 
    for label in xrange(self.n_labels):
      self.c_word_label[label] = [0] * self.N
      
    # generate structure for label assignments
    for doc_id, doc in enumerate(self.corpus):
      if random_topics:
        label = util.sample([1./self.n_labels] * self.n_labels, self.rng)
        self.label_assign[doc_id] = label
        self.change_label_counts(doc_id, doc, label, 1)
      else:
        self.label_assign[doc_id] = None
  
  def change_label_counts(self, doc_id, doc, label, c):
    for word in doc:
      self.c_word_label[label][word] += c
    self.c_label[(label)] += c

  def label_transition_probs(self, doc_id, doc):
    probs =  [0.0] * self.n_labels
    for label in xrange(self.n_labels):
      probs[label] += util.safe_log(self.c_label[label] + self.gamma_pi)
      for word in doc:
          probs[label] += util.safe_log(self.theta[label][word])
    if self.debug:
      print 'sample from probs:', probs

    return probs
  
  
  def sample_labels(self):
    for doc_id, doc in enumerate(self.corpus):
      current_label = self.label_assign[doc_id]

      # check whether label has been initialized
      if current_label != None:
        self.change_label_counts(doc_id, doc, current_label, -1)
      
      probs = self.label_transition_probs(doc_id, doc)
      new_label = util.sample_log(probs,self.rng)
      self.label_assign[doc_id] = new_label
      self.change_label_counts(doc_id, doc, new_label, 1)
    if self.debug:
      print 'n_label', self.c_label
      print 'n_word_label', self.c_word_label
      print 'labels:', self.label_assign
      
  
  def sample_theta(self):
    self.theta_hist.append(deepcopy(self.theta))
    
    for label in xrange(self.n_labels):
      dirichlet_params = util.v_sum([self.gamma_theta] * self.N, self.c_word_label[label])
      self.theta[label] = util.sample_dirichlet(dirichlet_params)
      if self.debug:
        print 'Dir-params:', dirichlet_params
        print 'theta:', self.theta[label]
     
  
  def gibbs_sampling(self):
    stderr.write("Starting sampling\n")
    self.print_parameters()
    for self.i in xrange(self.iterations):
      stderr.write('%d\r' % self.i)
      if self.debug:
        print
        print 'Iteration', self.i

      self.sample_theta()
      self.sample_labels()
   
  def __str__(self):
    s = ""
    for doc_id, doc in enumerate(self.corpus):
      s += "%d\t" % self.label_assign[doc_id]
      s += " ".join([self.corpus.vocabulary[i] for i in doc]) + "\n"
      
    return s
  
    
  def compute_average_theta(self,n_iter=None):
    self.avg_theta = [None] * self.n_labels
    norm = 0
    for theta in self.theta_hist[self.burn_in:n_iter:self.lag]:
      norm += 1
      for topic, probs in enumerate(theta):
        if topic not in self.avg_theta:
          self.avg_theta[topic] = probs
        else:
          self.avg_theta[topic] = util.v_sum(self.avg_theta[topic], probs)
    
    for probs in self.avg_theta:
      #self.avg_theta[topic] = [x/float(norm) for x in self.avg_theta[topic]]
      util.normalize(probs)
 
  def print_theta(self, theta):
    for topic, probs in enumerate(theta):
      if probs != None:
        print "Topic", topic
        for w,p in sorted(zip(self.corpus.vocabulary, probs), key=itemgetter(1), reverse=True):
          print '  %s\t%g' % (w,p)
 



if __name__ == "__main__":
  ### parameters
  n_labels = 2
  gamma_pi = 100./n_labels
  gamma_theta =.001
  burn_in = 10
  lag = 10
  iterations = 100
  
  ### load corpus
  c = Corpus('toy/toy1.dat', 'toy/vocab1.txt')

  ### build model
  m = Model(c, n_labels=n_labels, 
            gamma_pi=gamma_pi, gamma_theta=gamma_theta, 
            burn_in=burn_in, lag=lag, iterations=iterations, debug=False)  
  
  ### run sampling
  m.gibbs_sampling()
  m.print_theta(m.theta)
  
    

  
