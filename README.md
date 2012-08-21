Bayesian Naive Bayes (BNB)
==========================

This is an implementation of unsupervised Bayesian Naive Bayes with Gibbs sampling. Although it is efficient enough to be applied to real data it should not be viewed as a stable tool.

Background
----------
This type of unsupervised model was first used in 

<i>Ted Pedersen. 1997. <bf>Knowledge lean word sense disambiguation.</bf> In Proceedings of AAAI’97/IAAI’97.</i>


For a simple introduction to Gibbs sampling methods please refer to 

<i>Philip Resnik and Eric Hardisty. 2010. <bf>Gibbs sampling for the uninitiated.</bf> Technical report, University of Maryland.</i>


Requirements
------------
Numpy needs to be installed. The code was tested with Python 2.7.3 and Numpy 1.6.1.


Data
----

The data needs to be converted into the C-LDA format (<http://www.cs.princeton.edu/~blei/lda-c/>). We supply a very small toy dataset that is hopefully self-explanatory. Basically, each word is replaced by an integer. This leads to two files: 

 - A .dat file where each word and its frequency are listed. This is similar to the SVNlight format, only that the first entry in each line is the number of tokens in total.

 - A .vocab file that contains the words in order of their index. This means that there are no unassigned indexes.