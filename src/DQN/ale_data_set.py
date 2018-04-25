"""
This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import time
import theano

floatX = theano.config.floatX


class DataSet(object):
    """ Dataset class 
    A replay memory consisting of circular buffers for observed images,
    actions, and rewards.

    """
    def __init__(self, width, height, u_width, rng, max_steps=1000, phi_length=4):
        """Construct a DataSet.

        Arguments:
            width, height, u_width - image size
            max_steps - the number of time steps to store
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """
        # TODO: Specify capacity in number of state transitions, not
        # number of saved time steps.

        # Store arguments.
        self.width = width
        self.height = height
        self.u_width = u_width
        self.max_steps = max_steps
        self.phi_length = phi_length
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((max_steps, height, width), dtype='float32')
        self.u_imgs = np.zeros((max_steps, height, u_width), dtype='float32')
        self.actions = np.zeros(max_steps, dtype='int32')
        self.responses = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, u_img, action, response, reward, terminal):
        """Add a time step record.

        Arguments:
            img -- observed image
            u_img -- obserbed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.imgs[self.top] = img
        self.u_imgs[self.top] = u_img
        self.actions[self.top] = action
        self.responses[self.top] = response
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.imgs.take(indexes, axis=0, mode='wrap'), self.u_imgs.take(indexes, axis=0, mode='wrap')

    def phi(self, img, u_img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
        u_phi = np.empty((self.phi_length, self.height, self.u_width), dtype=floatX)
        phi[0:self.phi_length - 1] = self.imgs.take(indexes, axis=0, mode='wrap')
        phi[-1] = img
        u_phi[0:self.phi_length - 1] = self.u_imgs.take(indexes, axis=0, mode='wrap')
        u_phi[-1] = u_img
        return phi, u_phi

    def random_batch(self, batch_size):
        """Return corresponding states, actions, responses, rewards, terminal status, and
next_states for batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        states = np.zeros((batch_size, self.phi_length, self.height, self.width), dtype='float32')
        u_states = np.zeros((4, batch_size, self.phi_length, self.height, self.u_width), dtype='float32')
        actions = np.zeros((batch_size, 1), dtype='int32')
        responses = np.zeros((4, batch_size, 1), dtype='int32')
        rewards = np.zeros((5, batch_size, 1), dtype=floatX)
        terminal = np.zeros((5, batch_size, 1), dtype='bool')
        next_states = np.zeros((batch_size, self.phi_length, self.height, self.width), dtype='float32')
        next_u_states = np.zeros((4, batch_size, self.phi_length, self.height, self.u_width), dtype='float32')

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            initial_indices = np.arange(index, index + self.phi_length)
            transition_indices = initial_indices + 1
            end_index = index + self.phi_length - 1

            if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
                continue

            # Add the state transition to the response.
            states[count] = self.imgs.take(initial_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[0][count] = self.rewards.take(end_index, mode='wrap')
            terminal[0][count] = self.terminal.take(end_index, mode='wrap')
            next_states[count] = self.imgs.take(transition_indices,
                                                axis=0,
                                                mode='wrap')
            count += 1
        
        for i in range(4):
            count = 0
            while count < batch_size:
                # Randomly choose a time step (action == 0) from the replay memory.
                index = np.random.choice(np.where(self.actions == i)[0])

                initial_indices = np.arange(index, index + self.phi_length)
                transition_indices = initial_indices + 1
                end_index = index + self.phi_length - 1

                if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
                    continue

                # Add the state transition to the response.
                u_states[i][count] = self.u_imgs.take(initial_indices, axis=0, mode='wrap')
                responses[i][count] = self.responses.take(end_index, mode='wrap')
                rewards[i + 1][count] = self.rewards.take(end_index, mode='wrap')
                terminal[i + 1][count] = self.terminal.take(end_index, mode='wrap')
                next_u_states[i][count] = self.u_imgs.take(transition_indices,
                                                           axis=0,
                                                           mode='wrap')
                count += 1

        return states, u_states, actions, responses, rewards, next_states, next_u_states, terminal


# TESTING CODE BELOW THIS POINT...

def simple_tests():
    np.random.seed(222)
    dataset = DataSet(width=2,
                      height=3,
                      u_width=2,
                      rng=np.random.RandomState(42),
                      max_steps=6, phi_length=4)
    for _ in range(10):
        img = np.random.randint(0, 256, size=(3, 2))
        u_img = np.random.randint(0, 256, size=(3, 2))
        action = np.random.randint(16)
        response = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        print 'img', img
        dataset.add_sample(img, u_img, action, response, reward, terminal)
        print "I", dataset.imgs
        print "A", dataset.actions
        print "RES", dataset.responses
        print "R", dataset.rewards
        print "T", dataset.terminal
        print "SIZE", dataset.size
        print
    print "LAST PHI", dataset.last_phi()
    print
    print 'BATCH', dataset.random_batch(2)


def speed_tests():

    dataset = DataSet(width=80,
                      height=80,
                      u_width=80,
                      rng=np.random.RandomState(42),
                      max_steps=20000, phi_length=4)

    img = np.random.randint(0, 256, size=(80, 80))
    u_img = np.random.randint(0, 256, size=(80, 80))
    action = np.random.randint(16)
    response = np.random.randint(16)
    reward = np.random.random()
    start = time.time()
    for _ in range(100000):
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset.add_sample(img, u_img, action, response, reward, terminal)
    print "samples per second: ", 100000 / (time.time() - start)

    start = time.time()
    for _ in range(200):
        a = dataset.random_batch(32)
    print "batches per second: ", 200 / (time.time() - start)

    print dataset.last_phi()


def trivial_tests():

    dataset = DataSet(width=2,
                      height=1,
                      u_width=2,
                      rng=np.random.RandomState(42),
                      max_steps=3, phi_length=2)

    img1 = np.array([[1, 1]], dtype='uint8')
    img2 = np.array([[2, 2]], dtype='uint8')
    img3 = np.array([[3, 3]], dtype='uint8')

    u_img1 = np.array([[1, 1]], dtype='uint8')
    u_img2 = np.array([[2, 2]], dtype='uint8')
    u_img3 = np.array([[3, 3]], dtype='uint8')

    dataset.add_sample(img1, u_img1, 1, 1, 1, False)
    dataset.add_sample(img2, u_img2, 2, 2, 2, False)
    dataset.add_sample(img3, u_img3, 2, 2, 2, True)
    print "last", dataset.last_phi()
    print "random", dataset.random_batch(1)


def max_size_tests():
    dataset1 = DataSet(width=3,
                       height=4,
                       u_width=3,
                       rng=np.random.RandomState(42),
                       max_steps=10, phi_length=4)
    dataset2 = DataSet(width=3,
                       height=4,
                       u_width=3,
                       rng=np.random.RandomState(42),
                       max_steps=1000, phi_length=4)
    for _ in range(100):
        img = np.random.randint(0, 256, size=(4, 3))
        u_img = np.random.randint(0, 256, size=(4, 3))
        action = np.random.randint(16)
        response = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset1.add_sample(img, u_img, action, response, reward, terminal)
        dataset2.add_sample(img, u_img, action, response, reward, terminal)
        np.testing.assert_array_almost_equal(dataset1.last_phi(),
                                             dataset2.last_phi())
        print "passed"


def test_memory_usage_ok():
    import memory_profiler
    dataset = DataSet(width=80,
                      height=80,
                      u_width=80,
                      rng=np.random.RandomState(42),
                      max_steps=100000, phi_length=4)
    last = time.time()

    for i in xrange(1000000000):
        if (i % 100000) == 0:
            print i
        dataset.add_sample(np.random.random((80, 80)), np.random.random((80, 80)), 1, 1, 1, False)
        if i > 200000:
            states, u_states, actions, responses, rewards, next_states, next_u_states, terminals = dataset.random_batch(32)
        if (i % 10007) == 0:
            print time.time() - last
            mem_usage = memory_profiler.memory_usage(-1)
            print len(dataset), mem_usage
        last = time.time()


def main():
    speed_tests()
    test_memory_usage_ok()
    max_size_tests()
    simple_tests()


if __name__ == "__main__":
    main()
