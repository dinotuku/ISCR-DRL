import os

import numpy as np

from util import *
import reader


class SimulatedUser(object):
    """

        Simulates human response to retrieval machine

    """
    def __init__(self, data_dir, keyterm_thres, choose_random_topic, use_survey):
        self.data_dir = data_dir

        self.keyterm_thres = keyterm_thres
        self.choose_random_topic = choose_random_topic

        self.ans = None
        self.feat_len = 49

        # surveyed distribution
        self.use_survey = use_survey
        self.survey_probs = {'doc': 78.26,
                             'keyterm': 95.66,
                             'request_type': 22, # no function
                             'topic': 72.68}

    def __call__(self, query, ans, ans_index):
        # query and desired answer
        self.query = query.copy()
        self.ans = ans
        self.ret = None

        # Information for actions
        keyterm_dir = os.path.join(self.data_dir, 'keyterm')
        request_dir = os.path.join(self.data_dir, 'request')
        topic_dir = os.path.join(self.data_dir, 'lda')
        ranking_dir = os.path.join(self.data_dir, 'topicRanking')

        self.keytermlist = reader.readKeytermlist(keyterm_dir, query)
        self.requestlist = reader.readRequestlist(request_dir, self.ans)
        self.topiclist = reader.readTopicWords(topic_dir)
        # Since top ranking is pre-sorted according to query
        self.topicRanking = reader.readTopicList(ranking_dir, ans_index)[:5]

    def feedback(self, request, test_flag=False):
        ret = request['ret']
        action = request['action']
        response = request['response']

        params = {}
        params['action'] = action

        if ret:
            relevant_list = np.asarray([int(self.ans.has_key(item[0])) for item in ret[:self.feat_len]])
        else:
            relevant_list = np.zeros(self.feat_len)
        feature = relevant_list.reshape(1, self.feat_len)

        if action == 'firstpass':
            params['query'] = self.query
        elif action == 'doc':
            if self.use_survey:
                l = [item[0] for item in ret if self.ans.has_key(item[0])]
                # Work around for empty list
                if len(l) == 0:
                    doc = None
                elif len(l) == 1:
                    doc = l[0]
                else:
                    # Normal procedure
                    flag = np.random.uniform() * 100
                    if flag < self.survey_probs['doc']:
                        doc = l[0]
                    else:
                        doc = l[1]
            else:
                # doc = next( ( item[0] for item in ret if self.ans.has_key(item[0]) ), None )
                # use q_network
                l = [item[0] for item in ret if self.ans.has_key(item[0])]
                docs = [None, None, None, None]
                for i in range(len(l)):
                    if i >= 4:
                        break
                    docs[i] = l[i]
                doc = docs[response]

            params['doc'] = doc
            params['choices'] = docs

        elif action == 'keyterm':
            if len(self.keytermlist) is not 0:
                keyterm = self.keytermlist[0][0]

                # Determine relevancy with doc mdoels
                cnt = 0.
                for a in self.ans:
                    model_path = os.path.join(self.data_dir,'docmodel',reader.IndexToDocName(a))
                    if reader.readDocModel(model_path).has_key(keyterm):
                        cnt += 1.

                del self.keytermlist[0]

                isrel =  ( True if cnt/len(self.ans) > self.keyterm_thres else False )
                params['isrel'] = isrel

                # Flip relevancy with keyterm probability
                if self.use_survey:
                    flag = np.random.uniform() * 100.
                    if flag > self.survey_probs['keyterm']:
                        params['isrel'] = not isrel

                # use q_network
                probs = [100, 95, 90, 85]
                flag = np.random.uniform() * 100
                if flag > probs[response]:
                    params['isrel'] = not isrel

                params['keyterm'] = keyterm
                params['choices'] = isrel

        elif action == 'request':
            if self.use_survey:
                request_type = self.survey_probs['request_type']
                if request_type <= len(self.requestlist):
                    flag = np.random.randint(request_type)
                else:
                    flag = np.random.randint(len(self.requestlist))
                request = self.requestlist[flag][0]
                del self.requestlist[flag]
            else:
                # request = self.requestlist[0][0]
                # del self.requestlist[0]
                # use q_network
                requests = [None, None, None, None]
                for i in range(len(self.requestlist)):
                    if i >= 4:
                        break
                    requests[i] = self.requestlist[i][0]
                request = requests[response]
                if request:
                    del self.requestlist[response]

            params['request'] = request
            params['choices'] = requests

        elif action == 'topic':
            if self.use_survey: # Choose first or second topic using probability
                flag = np.random.uniform() * 100
                if flag < self.survey_probs['topic']:
                    topicIdx = self.topicRanking[0][0]
                    del self.topicRanking[0]
                else:
                    topicIdx = self.topicRanking[1][0]
                    del self.topicRanking[1]
            else:
                if self.choose_random_topic: # Choose random topic using topic weights
                    score = [ score*(score > 0) for _, score in self.topicRanking ]
                    if sum(score) == 0:
                        topicIdx = None
                    else:
                        prob = [s / sum(score) for s in score]
                        choice = np.random.choice(len(self.topicRanking), 1, p=prob)[0]
                        topicIdx = self.topicRanking[choice][0]
                        del self.topicRanking[choice]
                else: # Original, choose best topic
                    # topicIdx = self.topicRanking[0][0]
                    # del self.topicRanking[0]
                    # use q_network
                    topics = [None, None, None, None]
                    for i in range(len(self.topicRanking)):
                        if i >= 4:
                            break
                        topics[i] = self.topicRanking[i][0]
                    topic = topics[response]
                    if topic:
                        del self.topicRanking[response]

            params['topic'] = topic
            params['choices'] = topics

        else:
            raise ValueError("Action does not exist: {}".format(action))

        return params, feature

    def view(self, params):
        self.ret = params['ret']
