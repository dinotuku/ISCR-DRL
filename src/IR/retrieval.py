import os
import sys
import operator
import math
from .util import *
from .retmath import *
# from .copy import *
'''
def batchRetrieve(queries, background, inv_index, docleng, alpha):
    return [ retrieve(query, background, inv_index, docleng, alpha) for query in queries ]
#    retrieved = []
#    cnt = 1
#    for query in queries:
#	#print 'Retrieving query %d...' % cnt
#	cnt += 1
#	retrieved.append( retrieve(query, background, \
#		inv_index, docleng, alpha))
#    return retrieved

def retrieve(query, background, inv_index, docleng, alpha):
    result = {}
    for i in xrange(1,5048):
        result[i] = -9999

    for wordID, weight in query.items():
        existDoc = {}
	for docID, val in inv_index[wordID].items():
	    existDoc[docID] = 1
	    # smooth doc model by background
	    alpha_d = docleng[docID]/(docleng[docID]+alpha)
	    qryprob = weight
	    docprob = (1-alpha_d)*background[wordID]+alpha_d*val
	    #docprob = (val*docleng[docID]+mu*background[wordID])/ \
		    #	  (docleng[docID] + mu)
	    if result[docID]!=-9999:
		result[docID] += cross_entropy(qryprob,docprob)
	    else:
		result[docID] = cross_entropy(qryprob,docprob)
	
	for docID, val in result.items():
	    if not docID in existDoc and\
		    wordID in background:
		alpha_d = docleng[docID]/(docleng[docID]+alpha)
		qryprob = weight
	        docprob = (1-alpha_d)*background[wordID]
		if result[docID]!=-9999:
		   result[docID] += cross_entropy(qryprob,docprob)
		else:
		    result[docID] = cross_entropy(qryprob,docprob)

    sorted_ret = sorted(result.items(),\
	    key=operator.itemgetter(1),reverse=True)	
    return sorted_ret
'''


def retrieveCombination(query, negquery, background, inv_index, docleng, alpha, beta):
    result = {}
    for i in range(1, 5048, 1):
        result[i] = -9999
    for wordID, weight in query.items():
        existDoc = {}
        for docID, val in inv_index[wordID].items():
            existDoc[docID] = 1
            # smooth doc model by background
            alpha_d = docleng[docID] / (docleng[docID] + alpha)
            qryprob = weight
            docprob = (1 - alpha_d) * background[wordID] + alpha_d * val
            # docprob = (val*docleng[docID]+mu*background[wordID])/ \
            #	  (docleng[docID] + mu)
            if result[docID] != -9999:
                result[docID] += cross_entropy(qryprob, docprob)
            else:
                result[docID] = cross_entropy(qryprob, docprob)

        for docID, val in result.items():
            if not docID in existDoc and\
                    wordID in background:
                alpha_d = docleng[docID] / (docleng[docID] + alpha)
                qryprob = weight
                docprob = (1 - alpha_d) * background[wordID]
                if result[docID] != -9999:
                    result[docID] += cross_entropy(qryprob, docprob)
                else:
                    result[docID] = cross_entropy(qryprob, docprob)

    for wordID, weight in negquery.items():
        existDoc = {}
        for docID, val in inv_index[wordID].items():
            existDoc[docID] = 1
            # smooth doc model by background
            alpha_d = docleng[docID] / (docleng[docID] + alpha)
            qryprob = weight
            docprob = (1 - alpha_d) * background[wordID] + alpha_d * val
            if result[docID] != -9999:
                result[docID] -= beta * cross_entropy(qryprob, docprob)
            else:
                result[docID] = -1 * beta * cross_entropy(qryprob, docprob)

        for docID, val in result.items():
            if not docID in existDoc and\
                    wordID in background:
                alpha_d = docleng[docID] / (docleng[docID] + alpha)
                qryprob = weight
                docprob = (1 - alpha_d) * background[wordID]
                if result[docID] != -9999:
                    result[docID] -= beta * cross_entropy(qryprob, docprob)
                else:
                    result[docID] = -1 * beta * cross_entropy(qryprob, docprob)

    sorted_ret = sorted(result.items(),
                        key=operator.itemgetter(1), reverse=True)
    return sorted_ret


'''
def evalAP(ret,ans):
    AP = 0.0
    cnt = 0.0
    get = 0.0
    for docID, val in ret:
        cnt += 1.0
        if docID in ans:
            get += 1.0
            AP += float(get)/float(cnt)
    if len(ans)!=0:
        AP /= float(len(ans))
    return AP

def evalMAP(retrieved,answers,verbose=False):
    APs = [ evalAP(retrieved[i],answers[i]) for i in xrange(len(retrieved)) ]
    MAP = sum(APs)/len(APs)
    if verbose:
	    return MAP, APs
    else:
	    return MAP
	
'''
