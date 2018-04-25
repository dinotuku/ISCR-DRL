from collections import defaultdict
import operator
import os
import pickle


def load_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def readRequestlist(request_dir,fileIDs):
    requests = defaultdict(float)
    for fileID in fileIDs.keys():
        filepath = os.path.join(request_dir, str(fileID))
        if not os.path.isfile(filepath):
            continue
        with open(filepath, 'r') as fin:
            for line in fin.readlines():
                pair = line.split()
                requests[int(pair[0])] += float(pair[1])

    request_list = sorted(requests.iteritems(), key=operator.itemgetter(1), reverse=True)
    return request_list

data = load_from_pickle('../data/onebest_CMVN/query.pickle')
ans = data[127][1]
print readRequestlist('../data/onebest_CMVN/request', ans)[:5]
