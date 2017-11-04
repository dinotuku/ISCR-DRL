import logging
import operator

from .retmath import cross_entropy
from .reader import *

class SearchEngine(object):
  def __init__(self,lex_file, background_file, inv_index_file, doclengs_file,alpha=1000,beta=0.1):
    # Initialize
    self.lex        = readLex(lex_file)
    self.background = readBackground(background_file,self.lex)
    self.inv_index  = readInvIndex(inv_index_file)
    self.doclengs   = readDocLength(doclengs_file)

    # Query expansion parameters
    self.alpha = alpha
    self.beta = beta

  def retrieve(self, query, negquery = None):
    """
      Retrieves result using query and negquery if negquery exists
    """
    result = {}
    for i in range(1,5048,1):
      result[i] = -9999
    # Query
    for wordID, weight in query.items():
      existDoc = {}
      for docID, val in self.inv_index[wordID].items():
        existDoc[docID] = 1

        # smooth doc model by background
        alpha_d = self.doclengs[docID]/(self.doclengs[docID]+self.alpha)
        qryprob = weight
        docprob = (1-alpha_d)*self.background[wordID]+alpha_d*val

        # Adds to result
        if result[docID] != -9999:
          result[docID] += cross_entropy(qryprob,docprob)
        else:
          result[docID] = cross_entropy(qryprob,docprob)

      # Run background model
      for docID, val in result.items():
        if not docID in existDoc and wordID in self.background:
          alpha_d = self.doclengs[docID] / ( self.doclengs[docID] + self.alpha )
          qryprob = weight
          docprob = (1-alpha_d) * self.background[wordID]
          if result[docID] != -9999:
            result[docID] += cross_entropy(qryprob,docprob)
          else:
            result[docID] = cross_entropy(qryprob,docprob)

    # Run through negative query
    if negquery:
      for wordID, weight in negquery.items():
        existDoc = {}
        for docID, val in self.inv_index[wordID].items():
          existDoc[docID] = 1
          # smooth doc model by background
          alpha_d = self.doclengs[docID]/(self.doclengs[docID]+self.alpha)
          qryprob = weight
          docprob = (1-alpha_d)*self.background[wordID]+alpha_d*val

          if result[docID] != -9999:
            result[docID] -= self.beta * cross_entropy(qryprob,docprob)
          else:
            result[docID] = -1 * self.beta * cross_entropy(qryprob,docprob)

        # Run through background model
        for docID, val in result.items():
          if not docID in existDoc and wordID in self.background:
            alpha_d = self.doclengs[docID]/(self.doclengs[docID]+self.alpha)
            qryprob = weight
            docprob = (1-alpha_d) * self.background[wordID]

          if result[docID] != -9999:
            result[docID] -= self.beta * cross_entropy(qryprob,docprob)
          else:
            result[docID] = -1 * self.beta * cross_entropy(qryprob,docprob)

    sorted_ret = sorted(result.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_ret
