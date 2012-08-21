from __future__ import division
from math import exp, log
import random


# creates a close-to-uniform probability vector
def randInit(k,rng):

    vector = []

    for i in range(k):
        f = 1 + rng.random()
        vector.append(f)
    normalize(vector)
    return vector

def v_sum(v1,v2):
  return [a + b for a,b in zip(v1, v2)]

# normalizes a probability vector to 1
def normalize(vector):

    # calc sum
    total = sum(vector)

    if total < 1e-99:
        # if the sum is to small ie approx. 0
        # return a new uniform vector
        for i,f in enumerate(vector):
            vector[i] = 1. / len(vector)
    else:
        for i,f in enumerate(vector):
            vector[i] /= total

# sets the vector to zero
def setToZero(vector):

    for i,f in enumerate(vector):
        vector[i] = 0.0


def sample(pdist,rng):
    normalize(pdist)
    r = rng.random()

    i = -1
    s = 0
    while s < r:
        i += 1
        s += pdist[i]
    
    return i


def sample_dirichlet(params):
  sample = [random.gammavariate(a,1) for a in params]
  normalize(sample)
  return sample    
    
def sample_log(log_pdist,rng):
  max_p = max(log_pdist)
  norm_log_pdist = [p - max_p for p in log_pdist]
  pdist = [exp(p) for p in norm_log_pdist]
  normalize(pdist)
  return sample(pdist, rng)

def reduce2(l, i1, i2, op):
  l[i1] = op(l[i1], l[i2])
  del l[i2]


def safe_log(l):
     # laplacian smoothing with very small value
     return log(l + 10e-250)
     
class Indexer:
  def __init__(self, initial_value=1):
    self.index_dict = {}
    self.max_index  = initial_value
  
  def __call__(self, w):
    return self.get_index(w)
  
  def get_vocabulary(self):
    return map(itemgetter(0), sorted(self.index_dict.items(), key=itemgetter(1)))
  
  def get_index(self, w):
    if w in self.index_dict:
      return self.index_dict[w]
    else:
      self.index_dict[w] = self.max_index
      r = self.max_index
      self.max_index += 1

      return r
      
  def __str__(self):
    return '\n'.join(self.get_vocabulary())
    
  def write_file(self, file_name):
    f = open(file_name, 'w')
    f.write(str(self))
    f.close()
