# Policy Learning by Weighting Exploration with the Returns
from math import cos, sin, tan, sqrt, exp

import numpy as np
import random
from sklearn.metrics import mutual_info_score
import math


def rollouts(states):
    '''Execute trajectories in environment, return joint configs'''
    return states


def execute_trajectories(T, theta, epsilon, phi, num_actions, num_states):

    '''Execute some amount of trajectories in a while loop'''

    # Dummy code below:
    a = np.zeros(num_actions)
    s = np.zeros(num_states)
    rewards = np.zeros(num_actions, T)
    t = np.arange(T)

    actions = []

    '''For each trajectory, execute this: '''
    for t in range(T):
        a[t] = (theta + epsilon[t])**T * phi

    '''Collect all the rewards and time'''

    return t, s, a, np.delete(s, 0, 0), epsilon, rewards


def kl_divergence(new_path_dist, weighted_path_dist):
    new_path_dist = np.asarray(new_path_dist, dtype=float)
    weighted_path_dist = np.asarray(weighted_path_dist, dtype=float)

    return np.sum(np.where(new_path_dist != 0, new_path_dist * np.log(new_path_dist / weighted_path_dist), 0))


def PoWER(states, states_count, action_count):

    def RBF(s, c):
        return np.average(exp(-2 * (s - c)**2))

    ''' For finite horizons of length T '''
    T = 10      # Change T if necessary

    # The initial states (joint angles in degrees), last one is release (1) or don't release (0)
    # Looks like this:
    # states = [0., 0., 0., 0., 0., 0., 0]
    states = np.tile(states, (states_count, 1))

    '''Initialize other stuff'''
    sigma = 1.0   # I chose this arbitrarily
    sigma_ij = sigma*np.ones([states_count, action_count])
    eps = np.reshape(np.random.normal(0, sigma_ij.flatten()), [states_count, action_count])

    ''' 
        Initialize weights theta_0.
        I initialized it to a random distribution
    '''
    theta = np.random.random([states_count, action_count]) + eps

    '''Perform rollouts to get centers (joint config samples)'''
    centers = rollouts(states)

    ''' Calculate feature vector'''
    phi = RBF(states, centers)

    ''' Sample trajectories with policy a = (theta + epsilon_t) * phi(s,t) 
       where [e_t]_i,j is normally distributed with mean 0 and sigma^2_i,j 
       '''
    '''Execute some amount of trajectories in a while loop'''

    ''' Collect t, s_t, a_t, s_t+1, epsilon_t, and r_t+1 for all t = 1 to T+1 '''
    t, s, a, s_next, epsilon, r_next = execute_trajectories(T, theta, eps, phi, action_count, states_count)
    rewards = []

    ''' Unbiased estimate, Q-hat '''
    Q = np.sum(rewards)

    '''Q times noise'''
    Q_weighted = np.sum(rewards*epsilon, axis=0)

    '''Update policy'''
    theta = theta + Q_weighted/Q

    '''Break out of while loop when error is below a certain threshold'''
    error = np.sum((Q_weighted/Q)**2)

    return theta

