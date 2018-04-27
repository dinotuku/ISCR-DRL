import logging
import operator

import numpy as np

from retmath import cross_entropy
import reader


class SearchEngine(object):
    def __init__(self, lex_file, background_file, inv_index_file, doclengs_file, alpha=1000, beta=0.1):
        # Initialize
        self.lex = reader.readLex(lex_file)
        self.background = reader.readBackground(background_file, self.lex)
        self.inv_index = reader.readInvIndex(inv_index_file)
        self.doclengs = reader.readDocLength(doclengs_file)

        # Query expansion parameters
        self.alpha = alpha
        self.beta = beta

    def retrieve(self, query, negquery=None):
        """
            Retrieves result using query and negquery if negquery exists
        """
        result = {}
        for i in range(0, len(self.doclengs), 1):
            result[i] = -9999
        # Query
        weight_history = []
        doc_val_history = []
        for wordID, weight in query.iteritems():
            weight_history.append(weight)
            existDoc = {}
            for docID, val in self.inv_index[wordID].iteritems():
                doc_val_history.append(val)
                existDoc[docID] = 1
                # smooth doc model by background
                alpha_d = self.doclengs[docID] / (self.doclengs[docID] + self.alpha)
                docprob = (1 - alpha_d) * self.background[wordID] + alpha_d * val
                qryprob = weight

                # Adds to result
                if result[docID] != -9999:
                    result[docID] += cross_entropy(docprob * qryprob)
                else:
                    result[docID] = cross_entropy(docprob * qryprob)

            # Run background model
            for docID, val in result.iteritems():
                if not existDoc.has_key(docID) and self.background.has_key(wordID):
                    doc_val_history.append(val)
                    alpha_d = self.doclengs[docID] / (self.doclengs[docID] + self.alpha)
                    docprob = (1 - alpha_d) * self.background[wordID]
                    qryprob = weight
                    if result[docID] != -9999:
                        result[docID] += cross_entropy(docprob * qryprob)
                    else:
                        result[docID] = cross_entropy(docprob * qryprob)

        # Run through negative query
        if negquery:
            for wordID, weight in negquery.iteritems():
                existDoc = {}
                for docID, val in self.inv_index[wordID].iteritems():
                    existDoc[docID] = 1
                    # smooth doc model by background
                    alpha_d = self.doclengs[docID] / (self.doclengs[docID] + self.alpha)
                    docprob = (1 - alpha_d) * self.background[wordID] + alpha_d * val
                    qryprob = weight
                    if result[docID] != -9999:
                        result[docID] -= self.beta * cross_entropy(docprob * qryprob)
                    else:
                        result[docID] = -1 * self.beta * cross_entropy(docprob * qryprob)

                for docID, val in result.iteritems():
                    if not existDoc.has_key(docID) and self.background.has_key(wordID):
                        alpha_d = self.doclengs[docID] / (self.doclengs[docID] + self.alpha)
                        docprob = (1 - alpha_d) * self.background[wordID]
                        qryprob = weight
                        if result[docID] != -9999:
                            result[docID] -= self.beta * cross_entropy(docprob * qryprob)
                        else:
                            result[docID] = -1 * self.beta * cross_entropy(docprob * qryprob)

        sorted_ret = sorted(result.iteritems(), key=operator.itemgetter(1),reverse=True)	
        return sorted_ret
