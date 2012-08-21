Bayesian Naive Bayes (BNB)
==========================

This is an implementation of unsupervised Bayesian Naive Bayes with Gibbs sampling.

Background
----------
This type of unsupervised model was first used in 

Ted Pedersen. 1997. Knowledge lean word sense disambiguation. In Proceedings of AAAI’97/IAAI’97.


For a simple introduction to Gibbs sampling methods please refer to 

Philip Resnik and Eric Hardisty. 2010. Gibbs sampling for the uninitiated. Technical report, University of Maryland.


Requirements
------------
Numpy needs to be installed. The code was tested with Python 2.7.3 and Numpy 1.6.1.


Data
----

The data needs to be converted into the C-LDA format (<http://www.cs.princeton.edu/~blei/lda-c/>). We supply a very small toy dataset that is hopefully self-explanatory.