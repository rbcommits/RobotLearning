# Policy Learning by Weighting Exploration with the Returns
from math import cos, sin, tan, sqrt, exp

import numpy as np
from environment import BallGame
import random
from sklearn.metrics import mutual_info_score
import math


'''Ignore 

def kl_divergence(new_path_dist, weighted_path_dist):
    new_path_dist = np.asarray(new_path_dist, dtype=float)
    weighted_path_dist = np.asarray(weighted_path_dist, dtype=float)

    return np.sum(np.where(new_path_dist != 0, new_path_dist * np.log(new_path_dist / weighted_path_dist), 0))

'''


def PoWER(states_count=12, action_count=256):

    def RBF(s, c):
        return np.sum(np.exp(-2 * (s - c)**2), axis=0)/float(s.shape[0])   # feature vector

    game = BallGame()
    game.new_episode()

    # The initial state (joint angles in degrees), last one is release (1) or don't release (0)
    # Looks like: state = [0., 0., 0., 0., 0., 0., 0]
    current_state = game.get_state()
    print(current_state)
    states = np.tile(current_state, (states_count, 1))


    '''Initialize other stuff'''
    sigma = 1.0   # I chose this arbitrarily
    sigma_ij = sigma*np.ones([states_count, action_count])
    eps = np.reshape(np.random.normal(0, sigma_ij.flatten()), [action_count, states_count])

    ''' 
        Initialize weights theta_0 to a random distribution
    '''
    theta = np.random.random([action_count, states_count]) + eps

    '''Perform rollouts to get centers (joint config samples)'''
    ''' For finite horizons of length T, defined in BallGame() or until ball is released '''
    centers=[]
    print((np.array(centers)).shape[0])
    print(len(centers) < states_count)
    test = (len(centers) < states_count)
    while len(centers) < states_count:
        while not game.is_episode_finished():
            phi = np.ones(states_count)     # Initially, the feature vector is RBF(0) = 1 everywhere
            action = np.dot(theta, phi)
            r = game.make_action(action)
            centers.append(game.get_state())
        game.new_episode()

    ''' Sample trajectories with policy a = (theta + epsilon) * phi(s) 
       where [e_t]_i,j is normally distributed with mean 0 and sigma^2_i,j 
       '''
    '''Execute trajectories in a while loop'''
    error = 1000
    final_rewards = []
    errors = []
    while error > 0.001:

        game.new_episode()
        episode_rewards = []
        ''' Calculate feature vector using centers '''
        phi = RBF(states, centers)

        t = 0
        while not game.is_episode_finished():
            action = np.dot(theta, phi)
            reward = game.make_action(action)
            episode_rewards.append(reward)
            current_state = game.get_state()
            t += 1

        states = np.tile(current_state, (states_count, 1))

        '''Adjust dimensions of noise according to reward '''
        # padding = eps[trajectory_counter].shape - episode_rewards.shape
        epsilon = eps[t]
        if len(episode_rewards) != eps[t].shape:
            epsilon = eps[t][:len(episode_rewards)]


        ''' Unbiased estimate, Q-hat '''
        Q = np.sum(episode_rewards)

        '''Q times noise'''
        Q_weighted = np.sum(episode_rewards*epsilon, axis=0)

        '''Update policy params'''
        theta = theta + Q_weighted/Q

        '''Break out of while loop when error is below a certain threshold'''
        error = np.sum((Q_weighted/Q)**2)

        errors.append(error)
        final_rewards.append(max(episode_rewards))


    np.save("final_rewards", final_rewards)
    np.save("errors", errors)
    print(final_rewards)

    return theta, eps, final_rewards, errors