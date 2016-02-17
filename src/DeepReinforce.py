import cPickle as pickle
import datetime
import logging
import os
import pdb
import random
import time
from termcolor import cprint
print_red = lambda x: cprint(x, 'red')
print_blue = lambda x: cprint(x, 'blue')
print_yellow = lambda x: cprint(x, 'yellow')
import numpy as np
import progressbar
from progressbar import ProgressBar, Percentage, Bar, ETA

from DQN import q_network
import DQN.agent as agent
from IR.environment import *
from IR.util import readFoldQueries,readLex,readInvIndex

##########################
#       filename         #
##########################

train_data = 'train.fold1.pkl'
test_data  = 'test.fold1.pkl'

dir='../../ISDR-CMDP/'
data_dir = '10fold/query/CMVN'

lex = 'PTV.lex'
background = 'background/onebest.CMVN.bg'
inv_index = 'index/onebest/PTV.onebest.CMVN.index'
doclengs = 'doclength/onebest.CMVN.length'
answers = 'PTV.ans'

docmodeldir = 'docmodel/onebest/CMVN/'

newdir = '../Data/query/'

training_data = pickle.load(open(newdir+train_data,'r'))
testing_data  = pickle.load(open(newdir+test_data,'r'))

def list2tuple(data):
  result = []
  for idx in range(len(data[0])):
    result.append(tuple( (data[0][idx],data[1][idx],data[2][idx]) ))
  return result

training_data = list2tuple(training_data)
testing_data  = list2tuple(testing_data)

###############################
input_width, input_height = [89,1]
num_actions = 5

phi_length = 1 # phi length?  input 4 frames at once num_frames
discount = 1.
learning_rate = 0.00025
rms_decay = 0.99 # rms decay
rms_epsilon = 0.1
momentum = 0
clip_delta = 0.
freeze_interval = 100 #???  no freeze?
batch_size = 32
network_type = 'rl_dnn'
update_rule = 'deepmind_rmsprop' # need update
batch_accumulator = 'sum'
rng = np.random.RandomState()
###############################
epsilon_start = 1.0
epsilon_min = 0.1
replay_memory_size = 10000
experiment_prefix = 'result/ret'
replay_start_size = 500
update_frequency = 1
###############################
num_epoch = 20

epsilon_decay = num_epoch * 500

test_frequency = 1

step_per_epoch = 1000
max_steps = 5

num_tr_query = len(training_data)
num_tx_query = len(testing_data)
print "number of trainig data: ", num_tr_query
print "number of testing data: ", num_tx_query
# TODO
# map -> ap  -- done
# action 2   -- done
# count num_steps -- done
# testing MAP,ans -- done (no state estimate)
# test print AP -- done (termcolor)
# test no random --  done (epsilon = 0,phi = 1,init episode no random)
# check 4 baseline
# check dict copy?
# overfit one query
# mix tr tx
# test progress bar
# simulate platform
###############################
# logging
exp_log_root = '../logs/'
try:
  os.makedirs(exp_log_root)
except:
  pass
cur_datetime = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
exp_log_name = exp_log_root + cur_datetime + ".log"
logging.basicConfig(filename=exp_log_name,level=logging.INFO)
###############################
class experiment():
  def __init__(self,agent,env):
    self.agent = agent
    self.env = env

  def run(self):
    for epoch in xrange(num_epoch):
      logging.info('epoch {0}'.format(epoch))
      print_red( 'Running epoch {0}'.format(epoch+1))
      ## TRAIN ##
      self.run_epoch()
      self.agent.finish_epoch(epoch+1)
      
      ## TEST ##
      self.agent.start_testing()
      self.run_epoch(True)
      self.agent.finish_testing(epoch+1)

  def run_epoch(self,test_flag=False):
    ## PROGRESS BAR SETTING
    title = ['Training', 'Testing']
    widgets = [ title[test_flag], Percentage(), Bar(), ETA() ]
    pbar = ProgressBar(widgets=widgets,maxval=step_per_epoch).start()

    steps_left = step_per_epoch
    while steps_left > 0:
      if True:
      #for q, ans, ans_index in training_data:
        q, ans, ans_index = training_data[0]
        logging.info( 'ans_index {0}'.format(ans_index) )
        n_steps = self.run_episode(q,ans,ans_index,test_flag)
        pbar.update(step_per_epoch - steps_left)
        steps_left -= n_steps
      if test_flag:
        print_yellow( 'Episode Reward : '+ str(self.agent.episode_reward) )
        break
    pbar.finish()

  def run_episode(self,q,ans,ans_index,test_flag = False):
    init_state = self.env.setSession(q,ans,ans_index,test_flag)  # Reset & First-pass
    action     = self.agent.start_episode(init_state)

    num_steps = 0
    while True:
      reward, state = self.env.step(action)				# ENVIROMENT STEP
      terminal, AP = self.env.game_over()
      if test_flag and action != 4:
        AM = self.env.dialoguemanager.actionmanager
        print 'action : ', action,' ',AM.actionTable[ action],'\tcost : ', AM.costTable[ action ] ,"\tAP : ", AP, "\treward : ", reward
      num_steps += 1
      if num_steps >= max_steps or terminal:  # STOP Retrieve
        self.agent.end_episode(reward, terminal)
        break
      action = self.agent.step(reward, state)			# AGENT STEP
    return num_steps

def launch():
  t = time.time()
  print 'Compiling Network...'
  network = q_network.DeepQLearner(input_width, input_height, num_actions,
                                         phi_length,
                                         discount,
                                         learning_rate,
                                         rms_decay,
                                         rms_epsilon,
                                         momentum,
                                         clip_delta,
                                         freeze_interval,
                                         batch_size,
                                         network_type,
                                         update_rule,
                                         batch_accumulator,
                                         rng)
  print 'Creating Agent and Simulator...'
  agt = agent.NeuralAgent(network,epsilon_start,epsilon_min,epsilon_decay,
                                  replay_memory_size,
                                  experiment_prefix,
                                  replay_start_size,
                                  update_frequency,
                                  rng)

  print 'Creating Environment and compiling State Estimator...'
  env = Environment(lex,background,inv_index,\
                    doclengs,docmodeldir,dir)
  print 'Initializing experiment...'
  exp = experiment(agt,env)
  print 'Done, time taken {} seconds'.format(time.time()-t)
  exp.run()

if __name__ == "__main__":
  launch()
