#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday 10 Sept, 2019
@author: PabloAMC
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from gym.envs.toy_text import discrete
import time
import timeit
import random
import seaborn as sns
from matplotlib.patches import Arrow


# Parameters
tam = 2
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
HOLES = {2}
GOAL = {3}
SLP = 0


# Environment
class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[2, 2]):
        
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape) # number of states
        nA = 4              # number of actions
        slp = 100 - SLP
        
                self.MAX_Y = shape[0]    # shape of the gridworld, x direction
        self.MAX_X = shape[1]    # shape of the gridworld, y direction

        P = {}              # dictionary of [states] [actions] = ([1.0, next_state, reward, is_done])
        self.grid = np.zeros(shape) - 1.0
        it = np.nditer(self.grid, flags=['multi_index']) # iteration over array 'grid'
        
        '''
        Numeration of the matrix 4x4 is as follows:
        0 1 2 3
        4 5 6 7
        8 9 10 11
        12 23 14 15
        '''

        while not it.finished:
            s = it.iterindex                    # states
            y, x = it.multi_index

            if s == (nS - 1):
                self.grid[y][x] = 0.0

            P[s] = {a : [] for a in range(nA)}  # dictionary with info of state, action and reward

            is_done = lambda s: s in GOAL #can be modified to include more goals
            is_dead = lambda s: s in HOLES #can be modified to include more holes
            reward = 1.0 if is_done(s) else -1.0 if is_dead(s) else 0.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            #else:           #One may want to include some kind of list of goal states to substitute the next four lines.
            ns_up = s if y == 0 else s - self.MAX_X # move one full row to the left
            ns_right = s if x == (self.MAX_X - 1) else s + 1
            ns_down = s if y == (self.MAX_Y - 1) else s + self.MAX_X # move one full row to the right
            ns_left = s if x == 0 else s - 1
            
            def rand_choice(orig):
                r = random.randint(0,3)
                if r == 0:
                    return ns_up
                elif r == 1:
                    return ns_right
                elif r == 2:
                    return ns_down
                elif r == 3:
                    return ns_left
                else:
                    return orig
            
            P[s][UP] = [(1.0, ns_up if random.randint(0,100) < slp else rand_choice(ns_up), reward, False)]
            P[s][RIGHT] = [(1.0, ns_right if random.randint(0,100) < slp else rand_choice(ns_right), reward, False)]
            P[s][DOWN] = [(1.0, ns_down if random.randint(0,100) < slp else rand_choice(ns_down), reward, False)]
            P[s][LEFT] = [(1.0, ns_left if random.randint(0,100) < slp else rand_choice(ns_left), reward, False)]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.zeros(nS)
        isd[0] = 1
        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)
        
# Utilities
def trans_mat(env):
    return (np.array([[np.eye(1, env.nS, env.P[s][a][0][1])[0] for a in range(env.nA)] for s in range(env.nS)]),
            np.array([env.P[s][0][0][2] for s in range(env.nS)]))

def expected_utility(a, s, U, trans_probs):
    """The expected utility of doing a in state s, according to the MDP and U."""
    return sum([p * U[s1] for s1, p in enumerate(trans_probs[s, a, :])])
        
