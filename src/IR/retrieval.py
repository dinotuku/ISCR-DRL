import os
import sys
import operator
import math
from util import *
from retmath import *
from copy import *


def retrieveCombination(query, negquery, background, inv_index, docleng, alpha, beta):
    result = {}
    for i in range(0, len(docleng), 1):
        result[i] = -9999
        for wordID, weight in query.iteritems():
            existDoc = {}
            for docID, val in inv_index[wordID].iteritems():
                existDoc[docID] = 1
                # smooth doc model by background
                alpha_d = docleng[docID] / (docleng[docID] + alpha)
                docprob = (1 - alpha_d) * background[wordID] + alpha_d * val
                qryprob = weight
                if result[docID] != -9999:
                    result[docID] += cross_entropy(qryprob, docprob)
                else:
                    result[docID] = cross_entropy(qryprob, docprob)

            for docID, val in result.iteritems():
                if not existDoc.has_key(docID) and background.has_key(wordID):
                    alpha_d = docleng[docID] / (docleng[docID] + alpha)
                    docprob = (1 - alpha_d) * background[wordID]
                    qryprob = weight
                    if result[docID] != -9999:
                        result[docID] += cross_entropy(qryprob, docprob)
                    else:
                        result[docID] = cross_entropy(qryprob, docprob)

        for wordID, weight in negquery.iteritems():
            existDoc = {}
            for docID, val in inv_index[wordID].iteritems():
                existDoc[docID] = 1
                # smooth doc model by background
                alpha_d = docleng[docID] / (docleng[docID] + alpha)
                docprob = (1 - alpha_d) * background[wordID] + alpha_d * val
                qryprob = weight
                if result[docID] != -9999:
                    result[docID] -= beta * cross_entropy(qryprob, docprob)
                else:
                    result[docID] = -1 * beta * cross_entropy(qryprob, docprob)

            for docID, val in result.iteritems():
                if not existDoc.has_key(docID) and background.has_key(wordID):
                    alpha_d = docleng[docID] / (docleng[docID] + alpha)
                    docprob = (1 - alpha_d) * background[wordID]
                    qryprob = weight
                    if result[docID] != -9999:
                        result[docID] -= beta * cross_entropy(qryprob, docprob)
                    else:
                        result[docID] = -1 * beta * cross_entropy(qryprob, docprob)

    sorted_ret = sorted(result.iteritems(), key=operator.itemgetter(1),reverse=True)	
    return sorted_ret
