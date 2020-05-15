'''

'''

import sys
import numpy as np
import scipy.ndimage as nimg
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import collections

C_init_pos_center = None
C_init_pos_radii = None
C_init_pos_radii_multiplier = None
C_global_policy_path = None
C_global_policy_vars = None

C_new_policy_path = None


class World:
    def __init__(self):
        self.ws = 16  # half window size
        self.ws2 = self.ws * 2  # for scale 2
        self.state_size = np.array([self.ws2, self.ws2, self.ws2, 1])

    def set_volume(self, v, gt=None):  # v:numpy array of shape=[X,Y,Z], gt:[x,y,z]
        self.vol = v
        padded_vol = np.zeros(shape=self.vol.shape + self.state_size[:3] * 2, dtype=np.uint8)
        padded_vol[self.state_size[0]:self.state_size[0] + self.vol.shape[0],
        self.state_size[1]:self.state_size[1] + self.vol.shape[1],
        self.state_size[2]:self.state_size[2] + self.vol.shape[2]] = self.vol
        self.padded_vol = padded_vol
        self.gt = gt

    def set_position(self, pos):  # pos: [x,y,z]
        self.init_pos = pos
        self.pos = self.init_pos.copy()

    def get_state(self):
        x, y, z = self.pos[0], self.pos[1], self.pos[2]

        # increment offset because of padding
        x += self.state_size[0]
        y += self.state_size[1]
        z += self.state_size[2]
        state_scale_1 = self.padded_vol[x - self.ws:x + self.ws, y - self.ws:y + self.ws, z - self.ws:z + self.ws]
        return state_scale_1[..., np.newaxis]

    def get_reward(self, action, step=2):
        pos = self.pos.copy()
        if action == 0:
            pos[0] += step
        elif action == 1:
            pos[0] -= step
        elif action == 2:
            pos[1] += step
        elif action == 3:
            pos[1] -= step
        elif action == 4:
            pos[2] += step
        else:
            pos[2] -= step

        dist_now = np.sqrt(np.sum((pos - self.gt) ** 2))
        dist_prev = np.sqrt(np.sum((self.pos - self.gt) ** 2))

        reward = -1.0
        if dist_now < dist_prev:
            reward = 1.0
        if dist_now <= 4.0:
            reward = 2.0

        self.pos = pos
        return reward

    def move(self, action):
        reward = self.get_reward(action)
        if np.any(self.pos < 0) or np.any(self.pos >= self.vol.shape):
            self.pos = self.init_pos
            reward = -10.0
            next_state = self.get_state()
        else:
            next_state = self.get_state()
        return reward, next_state


class Agent:
    def __init__(self):
        self.env = World()
        self.build_model()

    def conv(self, layer_in, k, n, pool=True):
        layer = tf.layers.conv3d(inputs=layer_in, kernel_size=[k, k, k],
                                 filters=n, activation=tf.nn.relu, padding='same')
        if pool:
            layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[2, 2, 2], strides=[2, 2, 2])
        return layer

    def policy_func(self, layer_in, n1, n2):
        layer = tf.layers.dense(inputs=layer_in, units=n1, activation=tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units=n2, activation=None)
        layer = tf.nn.softmax(layer)
        layer = tf.clip_by_value(layer, 1e-3, 0.999)
        return layer

    def compute_loss(self, policys, actions, old_probs):
        actions_one_hot = tf.one_hot(indices=actions, depth=2, on_value=1.0, off_value=0.0)
        action_probs = tf.multiply(policys, actions_one_hot)
        action_probs = tf.reduce_sum(action_probs, axis=1)

        rt = action_probs / old_probs

        clipped_surrogate_loss = - tf.reduce_mean(
            tf.reduce_min([rt * self.advantage, tf.clip_by_value(rt, 0.8, 1.2) * self.advantage]))

        return clipped_surrogate_loss

    def build_model(self):

        self.input = tf.placeholder(shape=[None, self.env.state_size[0],
                                           self.env.state_size[1], self.env.state_size[2],
                                           self.env.state_size[3]], dtype=tf.float32)

        self.advantage = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)
        self.learning_rate = tf.placeholder(shape=None, dtype=tf.float32)
        self.pi_old = tf.placeholder(shape=[None, ], dtype=tf.float32)

        self.layer = self.conv(self.input, 3, 16)  # dim:16
        self.layer = self.conv(self.layer, 3, 16)  # dim:8
        self.layer = self.conv(self.layer, 3, 32)  # dim:4
        self.layer = self.conv(self.layer, 3, 32)  # dim:2
        self.layer = self.conv(self.layer, 3, 64)  # dim:1
        self.layer = tf.reshape(self.layer, shape=[-1, 64])

        # policy nets

        self.policy_x = self.policy_func(self.layer, 8, 2)
        self.policy_y = self.policy_func(self.layer, 8, 2)
        self.policy_z = self.policy_func(self.layer, 8, 2)

    def define_losses(self):
        # loss
        self.policy_loss_x = self.compute_loss(self.policy_x, self.actions, self.pi_old)
        self.policy_loss_y = self.compute_loss(self.policy_y, self.actions, self.pi_old)
        self.policy_loss_z = self.compute_loss(self.policy_z, self.actions, self.pi_old)

        # optimizers
        self.policy_opt_x = tf.train.AdamOptimizer(self.learning_rate).minimize(self.policy_loss_x)
        self.policy_opt_y = tf.train.AdamOptimizer(self.learning_rate).minimize(self.policy_loss_y)
        self.policy_opt_z = tf.train.AdamOptimizer(self.learning_rate).minimize(self.policy_loss_z)

        self.sess = tf.Session()

    def pi_x(self, state):
        if len(state.shape) == 4:
            state = state[np.newaxis, ...]
        return self.sess.run(self.policy_x, {self.input: state})

    def pi_y(self, state):
        if len(state.shape) == 4:
            state = state[np.newaxis, ...]
        return self.sess.run(self.policy_y, {self.input: state})

    def pi_z(self, state):
        if len(state.shape) == 4:
            state = state[np.newaxis, ...]
        return self.sess.run(self.policy_z, {self.input: state})

    def pi(self, axis, state):
        if axis == 0:
            return self.pi_x(state)
        elif axis == 1:
            return self.pi_y(state)
        else:
            return self.pi_z(state)

    def opt_pi_x(self, state, action, advantage, pi_old, alpha):
        self.sess.run(self.policy_opt_x, {self.input: state, self.actions: action, self.pi_old: pi_old,
                                          self.advantage: advantage,
                                          self.learning_rate: alpha})

    def opt_pi_y(self, state, action, advantage, pi_old, alpha):
        self.sess.run(self.policy_opt_y, {self.input: state, self.actions: action, self.pi_old: pi_old,
                                          self.advantage: advantage,
                                          self.learning_rate: alpha})

    def opt_pi_z(self, state, action, advantage, pi_old, alpha):
        self.sess.run(self.policy_opt_z, {self.input: state, self.actions: action, self.pi_old: pi_old,
                                          self.advantage: advantage,
                                          self.learning_rate: alpha})

    def opt_pi(self, axis, state, action, advantage, pi_old, alpha):
        if len(state.shape) == 4:
            state = state[np.newaxis, ...]
            action = [action]
            advantage = [advantage]
            pi_old = [pi_old]

        if axis == 0:
            self.opt_pi_x(state, action, advantage, pi_old, alpha)
        elif axis == 1:
            self.opt_pi_y(state, action, advantage, pi_old, alpha)
        else:
            self.opt_pi_z(state, action, advantage, pi_old, alpha)


# Create world and agent
world = World()
agent = Agent()

saver = tf.train.Saver()
agent.define_losses()
agent.sess.run(tf.global_variables_initializer())


# func: perform a single step in the world
def one_step_explore(world, agent, state, axis, epsilon=0.7):
    policy = np.squeeze(agent.pi(axis, state / 255.0))

    action = np.argmax(policy)
    random_action = np.random.randint(0, 2)
    if np.random.random() > epsilon:
        action = random_action

    reward, next_state = world.move(action + axis * 2)

    return state, action, reward, next_state, policy[action]


# func: run an episode in the world
def episode_explore(world, agent, max_step=150, init_pos=None, epsilon=0.7):
    pt, s_, a_, r_, p_, ax_ = [], [], [], [], [], []

    if init_pos is None:
        disparity = np.random.randint(-C_init_pos_radii, C_init_pos_radii + 1, [3])
        disparity *= C_init_pos_radii_multiplier
        init_pos = C_init_pos_center + disparity
        xx, yy, zz = init_pos[0], init_pos[1], init_pos[2]

    else:
        xx, yy, zz = init_pos[0], init_pos[1], init_pos[2]

    world.set_position(np.array([xx, yy, zz]))
    state = world.get_state()

    for step in range(max_step):
        to_break = False

        for ax in range(3):
            state, action, reward, next_state, policy = one_step_explore(world, agent, state, ax, epsilon)
            pt.append(world.pos)
            s_.append(state)
            a_.append(action)
            r_.append(reward)
            p_.append(policy)
            ax_.append(ax)

            if reward == -10:
                to_break = True
                break
            state = next_state

        if to_break:
            break

    return step, pt, s_, a_, r_, p_, ax_


# func: run N episodes in the world
def explore(world, agent, max_episode=8, max_step=50, init_pos=None, epsilon=0.7):
    step_, pt_, s_, a_, r_, p_, ax_ = [], [], [], [], [], [], []

    for episode in range(max_episode):
        step, pt, s, a, r, p, ax = episode_explore(world, agent, max_step=max_step, epsilon=epsilon, init_pos=init_pos)

        step_.append(step)
        s_.extend(s)
        a_.extend(a)
        r_.extend(r)
        p_.extend(p)
        ax_.extend(ax)

        if episode == 0:
            pt_.append(pt)
        else:
            pt_.append(pt)

    return step_, pt_, s_, a_, r_, p_, ax_


# func: runs N episodes, and returns mean results
def test(max_episode=10, max_step=50, epsilon=1.0):
    step, pt, s, a, r, p, ax = explore(world, agent, max_episode=max_episode, max_step=max_step, init_pos=None,
                                       epsilon=epsilon)
    mean_step = np.mean(step)
    last_pt = [pt[i][(step[i] - 1) * 3] for i in range(len(step))]  # step-1, to exclude the (probably) incomplete last step
    mean_pt = np.mean(last_pt, axis=0)
    var_pt = np.sum(np.var(last_pt, axis=0) ** 2)

    return mean_step, mean_pt, var_pt, s, a, r, p, ax


# func: given trajectories/experiences, perform one gradient step towards optimizing the policy
def update_policy(s, a, r, p, ax, alpha=1e-6, batch_size=20.0):
    s, a, r, p, ax = np.array(s), np.array(a), np.array(r), np.array(p), np.array(ax)
    s_x, s_y, s_z = s[ax == 0,], s[ax == 1,], s[ax == 2,]
    a_x, a_y, a_z = a[ax == 0,], a[ax == 1,], a[ax == 2,]
    r_x, r_y, r_z = r[ax == 0,], r[ax == 1,], r[ax == 2,]
    p_x, p_y, p_z = p[ax == 0,], p[ax == 1,], p[ax == 2,]

    n = len(r_z)

    # optimize the policy
    rand_id = np.random.permutation(n)
    for batch in range(np.int32(np.ceil(n / batch_size))):
        start, end = np.int32(batch * batch_size), np.int32((batch + 1) * batch_size)
        if end > n:
            end = n
        batch_mem = rand_id[start:end]
        agent.opt_pi(0, s_x[batch_mem,] / 255.0, a_x[batch_mem,], r_x[batch_mem,], p_x[batch_mem,], alpha)
        agent.opt_pi(1, s_y[batch_mem,] / 255.0, a_y[batch_mem,], r_y[batch_mem,], p_y[batch_mem,], alpha)
        agent.opt_pi(2, s_z[batch_mem,] / 255.0, a_z[batch_mem,], r_z[batch_mem,], p_z[batch_mem,], alpha)


# func: to train the policy, by gathering experiences, over a number of epochs
def train(max_episode=5, max_step=20, max_epoch=300, epsilon=0.1, alpha=1e-2, max_ppo_epoch=8, batch_size=20.0):
    global mean_rewards, mean_errors

    mean_rewards = []
    mean_errors = []
    plt.axis([0, max_epoch, -2.5, 2.5])
    plt.ion()
    plt.show()
    plt.xlabel('Epoch')
    plt.ylabel('Mean rewards per step')
    r_plt, = plt.plot(mean_rewards)

    # run initial explorations
    mean_step, mean_pt, var_pt, s, a, r, p, ax = test(max_episode, max_step, epsilon)
    best_error = 9999.0
    # epsilon = 0.1
    for epoch in range(max_epoch):
        epsilon += 1.0 / max_epoch
        if epsilon > 1.0: epsilon = 1.0

        # update policy for gathered experience
        for ppo_epoch in range(max_ppo_epoch):
            update_policy(s, a, r, p, ax, alpha, batch_size)

        # gather new experience
        mean_step, mean_pt, var_pt, s, a, r, p, ax = test(max_episode, max_step, epsilon)

        mean_reward = np.mean(r)
        mean_rewards.append(mean_reward)

        mean_error = np.sqrt(np.sum((world.gt - mean_pt) ** 2))
        mean_errors.append(mean_error)

        if best_error >= mean_error and epsilon == 1.0:
            best_error = mean_error
            saver.save(agent.sess, C_new_policy_path)

        print('epoch:', epoch, 'mean_error:', mean_error, 'mean_reward:', mean_reward, 'best_error:', best_error)

        r_plt.set_xdata(range(len(mean_rewards)))
        r_plt.set_ydata(mean_rewards)
        plt.draw()
        plt.pause(0.05)

    return best_error, mean_rewards, mean_errors


def fetch_arg(name):
    val = None
    try:
        idx = sys.argv.index(name)
        val = sys.argv[idx+1]
    except ValueError:
        val = None
    return val


if __name__ == '__main__':

    mode = fetch_arg('-mode')
    if mode is None:
        print('-mode not given.')
        sys.exit()
    print(mode + ' mode ->')

    print('loading volume...')
    volume_path = fetch_arg('-volume_path')
    if volume_path is None:
        print('-volume_path not given.')
        sys.exit()
    print('volume_path is:', volume_path)
    d = sio.loadmat(volume_path)

    print('setting volume to the world...')
    print('vol_shape:', d['vol'].shape, 'gt:', d['gt'])
    if len(d['gt'].shape)>1: d['gt'] = d['gt'][0]
    world.set_volume(v=d['vol'], gt=d['gt'])

    print(world.gt)

    print('fetching network_path...')
    network_path = fetch_arg('-network_path')
    if network_path is None:
        print('-network_path not given. default path set.')
        network_path = 'net/policy_best'
    C_new_policy_path = network_path
    print('network_path is:', C_new_policy_path)

    # set parameters
    max_episode = 5
    max_step = 30
    max_epoch = 300
    epsilon = float(0.7)
    alpha = float(1e-6)
    max_ppo_epoch = 2
    batch_size = float(20.0)

    # define init_pos sample space
    C_init_pos_center = np.array(world.vol.shape) // 2
    C_init_pos_radii = 5  # for exactly the center point (fixed initial state)
    C_init_pos_radii_multiplier = 1  # to expand the space with stride

    # set from args if given
    arg = fetch_arg('-max_episode')
    if arg: max_episode = int(arg)
    arg = fetch_arg('-max_step')
    if arg: max_step = int(arg)
    arg = fetch_arg('-max_epoch')
    if arg: max_epoch = int(arg)
    arg = fetch_arg('-epsilon')
    if arg: epsilon = float(arg)
    arg = fetch_arg('-alpha')
    if arg: alpha = float(arg)
    arg = fetch_arg('-batch_size')
    if arg: batch_size = float(arg)
    arg = fetch_arg('-max_ppo_epoch')
    if arg: max_ppo_epoch = int(arg)
    arg = fetch_arg('-init_pos_center')
    if arg: C_init_pos_center = np.array(eval(arg))
    arg = fetch_arg('-init_pos_radii')
    if arg: C_init_pos_radii = int(arg)
    arg = fetch_arg('-init_pos_radii_multiplier')
    if arg: C_init_pos_radii_multiplier = int(arg)

    if mode == 'train':
        best_error, mean_rewards, mean_errors = train(max_episode=max_episode, max_step=max_step, max_epoch=max_epoch,
                                                      epsilon=epsilon, alpha=alpha, max_ppo_epoch=max_ppo_epoch,
                                                      batch_size=batch_size)

    else:  # mode: test
        # restore the saved policy
        print('restoring saved policy from ' + C_new_policy_path + '...')
        saver.restore(agent.sess, C_new_policy_path)

        _, mean_pt, var_pt, _, _, _, _, _ = test(max_episode=max_episode, max_step=max_step, epsilon=epsilon)
        print('mean_pt:', mean_pt, 'variance:', var_pt)
