from sys import stderr
from collections import defaultdict
import numpy as np
from operator import itemgetter

class Corpus():
  def __init__(self, data_file=None, vocabulary_file=None, format='LDA'):
    stderr.write('Corpus file: %s\n' % data_file)
    self.vocabulary = []
    self.documents = []
    self.n_types = 0
    self.n_tokens = 0
    self.n_docs = 0
    self.format = format

    if data_file:
      stderr.write('Reading data\n')
      self.read_data(data_file, format=self.format)
  
    if vocabulary_file:
      stderr.write('Reading vocabulary\n')
      self.read_vocabulary(vocabulary_file)
      self.index_dict = dict([(v,k) for k,v in enumerate(self.vocabulary)])
      
  def read_data(self, file_name, format='LDA'):
    current_doc = []
    
    for line in open(file_name):
      line = line.strip()
      if line == "" or line.startswith('#'):
        continue
      tokens = line.split()
      
      if format == 'LDA':
        for token in tokens[1:]:
          t,f = map(int, token.split(':'))
          self.n_tokens += f
          t -= 1
          for i in xrange(f):
            current_doc.append(t)
      elif format == 'HBC':
        current_doc = [int(x)-1 for x in tokens]
      
      self.documents.append(current_doc)
      self.n_docs += 1
      current_doc = []

  def read_vocabulary(self, file_name):

    for line in open(file_name):
      line = line.strip()
      if line != "":
        self.vocabulary.append(line)
    self.n_types = len(self.vocabulary)

  def __iter__(self):
    return self.documents.__iter__()
  
  def __str__(self):
    s = ''
    for doc in self.documents:
      s += ' '.join([self.vocabulary[i] for i in doc]) + '\n'
    return s
  
      
  def word(self, doc_id, word_id):
    return self.documents[doc_id][word_id]
    
