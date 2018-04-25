"""
The NeuralAgent class wraps a deep Q-network for training and testing
in the Arcade learning environment.

Author: Nathan Sprague

"""
import os
import cPickle
import time
import logging

import numpy as np

import ale_data_set

import sys
sys.setrecursionlimit(10000)

class NeuralAgent(object):
    def __init__(self, u_network, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref,
                 replay_start_size, update_frequency, iterative_frequency, rng):

        self.u_network = u_network
        self.network = q_network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.exp_pref = exp_pref
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.iterative_frequency = iterative_frequency
        self.rng = rng

        self.iterative_indicator = True

        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height
        self.u_image_width = self.u_network[0].input_width

        # Create a folder to hold results
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                       "{}".format(self.network.lr).replace('.', 'p') + '_' + \
                       "{}".format(self.network.discount).replace('.', 'p')

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.num_actions = self.network.num_actions
        self.num_responses = self.u_network[0].num_actions

        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             u_width=self.u_image_width,
                                             rng=rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  u_width=self.u_image_width,
                                                  rng=rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        # self._open_results_file()
        # self._open_learning_file()

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_u_img = None
        self.last_action = None
        self.last_response = None

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,mean_u_loss_a0,mean_u_loss_a1,mean_u_loss_a2,mean_u_loss_a3,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                        self.total_reward / float(num_episodes),
                                        holdout_sum)
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{},{},{},{},{}\n".format(np.mean(self.loss_averages),
                                           np.mean(self.u_loss_a0_averages),
                                           np.mean(self.u_loss_a1_averages),
                                           np.mean(self.u_loss_a2_averages),
                                           np.mean(self.u_loss_a3_averages),
                                           self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def start_episode(self, observation, u_observation):
        """
            This method is called once at the beginning of each episode.
            No reward is provided, because reward is only available after
            an action has been taken.

            Arguments:
                observation - height x width numpy array
                u_observation - one dimension numpy array

            Returns:
                An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []
        self.u_loss_a0_averages = []
        self.u_loss_a1_averages = []
        self.u_loss_a2_averages = []
        self.u_loss_a3_averages = []

        self.start_time = time.time()

#        return_action = self.rng.randint(0, self.num_actions)
        if self.testing:
            phi, u_phi = self.test_data_set.phi(observation, u_observation)
            return_action = self.network.choose_action(phi, 0.05)
            if return_action == 0:
                return_response = self.u_network[0].choose_action(u_phi, 0.05)
            elif return_action == 1:
                return_response = self.u_network[1].choose_action(u_phi, 0.05)
            elif return_action == 2:
                return_response = self.u_network[2].choose_action(u_phi, 0.05)
            elif return_action == 3:
                return_response = self.u_network[3].choose_action(u_phi, 0.05)
        else:
            return_action = self.rng.randint(0, self.num_actions)
            return_response = self.rng.randint(0, self.num_responses)

        self.last_action = return_action
        self.last_response = return_response

        self.last_img = observation
        self.last_u_img = u_observation

        self.act_seq = [return_action]
        self.u_act_seq = [return_response]

        return return_action, return_response


    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

    def step(self, reward, observation, u_observation):
        """
        This method is called each time step.

        Arguments:
            reward      - Real valued reward.
            observation - A height x width numpy array
            u_observation - A one dimension numpy array

        Returns:
            An integer action.
        """

        self.step_counter += 1

        #TESTING---------------------------
        if self.testing:
            self.episode_reward += reward
            action, response = self._choose_action(self.test_data_set, 0.05,
                                                   observation,
                                                   u_observation,
                                                   np.clip(reward, -1, 1))

        #NOT TESTING---------------------------
        else:
            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)

                action, response = self._choose_action(self.data_set, self.epsilon,
                                                       observation,
                                                       u_observation,
                                                       np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
                    self.batch_counter += 1
                    if self.episode_counter % self.iterative_frequency == 0:
                        self.iterative_indicator = not self.iterative_indicator
                    if self.iterative_indicator:
                        loss = self._do_training()
                        self.loss_averages.append(loss)
                    else:
                        loss_a0, loss_a1, loss_a2, loss_a3 = self._do_training()
                        self.u_loss_a0_averages.append(loss_a0)
                        self.u_loss_a1_averages.append(loss_a1)
                        self.u_loss_a2_averages.append(loss_a2)
                        self.u_loss_a3_averages.append(loss_a3)
                    # loss = self._do_training()
                    # self.loss_averages.append(loss)

            else: # Still gathering initial random data...
                action, response = self._choose_action(self.data_set, self.epsilon,
                                                       observation,
                                                       u_observation,
                                                       np.clip(reward, -1, 1))

        self.last_action = action
        self.last_response = response
        self.last_img = observation
        self.last_u_img = u_observation
        self.act_seq.append(action)
        self.u_act_seq.append(response)

        return action, response

    def _choose_action(self, data_set, epsilon, cur_img, cur_u_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_u_img,
            self.last_action, self.last_response, reward, False)
        if self.step_counter >= self.phi_length:
            phi, u_phi = data_set.phi(cur_img, cur_u_img)
            action = self.network.choose_action(phi, epsilon)
            if action == 0:
                response = self.u_network[0].choose_action(u_phi, epsilon)
            elif action == 1:
                response = self.u_network[1].choose_action(u_phi, epsilon)
            elif action == 2:
                response = self.u_network[2].choose_action(u_phi, epsilon)
            elif action == 3:
                response = self.u_network[3].choose_action(u_phi, epsilon)
        else:
            # print('random action')
            action = self.rng.randint(0, self.num_actions)
            response = self.rng.randint(0, self.num_responses)

        return action, response

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        states, u_states, actions, responses, rewards, next_states, next_u_states, terminals = \
            self.data_set.random_batch(self.network.batch_size)
        
        if self.iterative_indicator:
            loss = self.network.train(states, actions, rewards[0],
                next_states, terminals[0])
            return loss
        else:
            loss_a0 = self.u_network[0].train(u_states[0], responses[0], rewards[1],
                next_u_states[0], terminals[1])
            loss_a1 = self.u_network[1].train(u_states[1], responses[1], rewards[2],
                next_u_states[1], terminals[2])
            loss_a2 = self.u_network[2].train(u_states[2], responses[2], rewards[3],
                next_u_states[2], terminals[3])
            loss_a3 = self.u_network[3].train(u_states[3], responses[3], rewards[4],
                next_u_states[3], terminals[4])
            return loss_a0, loss_a1, loss_a2, loss_a3

        # loss = self.network.train(states, actions, rewards[0],
        #     next_states, terminals[0])
        return loss

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.episode_loss = 0
        self.episode_u_loss_a0 = 0
        self.episode_u_loss_a1 = 0
        self.episode_u_loss_a2 = 0
        self.episode_u_loss_a3 = 0
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:
            self.episode_counter += 1
            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_u_img,
                                     self.last_action,
                                     self.last_response,
                                     np.clip(reward, -1, 1),
                                     True)

#            logging.info("steps/second: {:.2f}".format(\
#                            self.step_counter/total_time))

            if self.batch_counter > 0:
                #self._update_learning_file()
                self.episode_loss = np.mean(self.loss_averages) if self.loss_averages else 0
                self.episode_u_loss_a0 = np.mean(self.u_loss_a0_averages) if self.u_loss_a0_averages else 0
                self.episode_u_loss_a1 = np.mean(self.u_loss_a1_averages) if self.u_loss_a1_averages else 0
                self.episode_u_loss_a2 = np.mean(self.u_loss_a2_averages) if self.u_loss_a2_averages else 0
                self.episode_u_loss_a3 = np.mean(self.u_loss_a3_averages) if self.u_loss_a3_averages else 0
                logging.debug("average loss: {:.4f}".format(self.episode_loss))
                logging.debug("average u_loss_a0: {:.4f}".format(self.episode_u_loss_a0))
                logging.debug("average u_loss_a1: {:.4f}".format(self.episode_u_loss_a1))
                logging.debug("average u_loss_a2: {:.4f}".format(self.episode_u_loss_a2))
                logging.debug("average u_loss_a3: {:.4f}".format(self.episode_u_loss_a3))

    def finish_epoch(self, epoch):
        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.network, net_file, -1)
        net_file.close()

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            self.holdout_data = self.data_set.random_batch(holdout_size)[0]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i, ...]))


if __name__ == "__main__":
    pass
