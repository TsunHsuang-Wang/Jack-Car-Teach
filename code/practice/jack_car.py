# python 2/3 compatibility
from __future__ import division

import numpy as np
import os.path as osp
import time

import env

class JackCar(object):
    def __init__(self, use_precomputed=True, conf=None, precomputed_dict=None):
        '''
        FUNC: constructor of JackCar object, if use_precomputed=False, you need to specify conf,
              otherwise, you need to specify precomputed_dict
        Argument:
            use_precomputed: a bool to determine whether precomputed environment dynamics is used
            conf: a dictionary specifying configuration of Jack-Car-Rental problem with keys:
                  'max_move': maximum number of car moves overnight
                  'max_cars_A': maximum number of cars allowed to be stored at A
                  'max_cars_B': maximum number of cars allowed to be stored at B
                  'rent_price': revenue earned from renting a car
                  'move_cost': cost to move a car from A to B or from B to A
                  'gamma': discout factor of the MDP
            precomputed_dict: a dictionary the same as conf but with extra 2 keys:
                  'P_path': path to .npy file containing transition maxtrix of full environment dynamics
                  'R_path': path to .npy file containing expected return matrix of full environemnt dynamics
        '''
        # problem settings
        if use_precomputed:
            if precomputed_dict is None: 
                raise ValueError('if \'use_precomputed\' is set to True, you have to specify \'precomputed_dict\'')
            conf_path = osp.abspath(osp.expanduser(precomputed_dict['conf_path']))
            conf = np.load(conf_path).tolist() # may not be a good way to load data
            self._max_move = conf['max_move']
            self._max_cars_A = conf['max_cars_A']
            self._max_cars_B = conf['max_cars_B']
            self._rent_price = conf['rent_price']
            self._move_cost = conf['move_cost']
            self._gamma = conf['gamma']
            P_path = osp.abspath(osp.expanduser(precomputed_dict['P_path']))
            R_path = osp.abspath(osp.expanduser(precomputed_dict['R_path']))
        else:
            self._max_move = conf['max_move']
            self._max_cars_A = conf['max_cars_A']
            self._max_cars_B = conf['max_cars_B']
            self._rent_price = conf['rent_price']
            self._move_cost = conf['move_cost']
            self._gamma = conf['gamma']

        # number of possible actions, -max_move~max_move
        self._n_acts = 2*self._max_move + 1

        # size of states, which is the number of all combinations of possible number
        # of cars at A and B
        state_size = (self._max_cars_A+1) * (self._max_cars_B+1)

        # deterministic policy which is associated with number of cars in A and B,
        # i.e. a table ranging from -max_move~max_move, indicating how many cars
        # should be moved from A to B (negative moves = moving from B to A).
        self._policy = np.random.randint(low=-self._max_move, high=self._max_move, 
                                         size=((state_size)), dtype=np.int8)

        # state value function following current policy
        self._V = np.zeros((state_size), dtype=np.float32)

        # FULL environment dynamics
        PRfull_shape = tuple((self._n_acts, state_size, state_size))
        if use_precomputed:
            # load precomputed P_full and R_full
            self._P_full = np.load(P_path)
            self._R_full = np.load(R_path)
        else:
            # transition matrix associated with triplet (action, current_state, next_state)
            self._P_full = np.zeros(PRfull_shape, dtype=np.float32)
            # expected return associated with triplet (action, current_state, next_state)
            self._R_full = np.zeros(PRfull_shape, dtype=np.float32)
            # compute 
            self._build_env_dynamics()
        
        # transition matrix and expected return if we follow current policy. This is just
        # partial information of full environment dynamics, and is constructed by internal
        # function _get_PR_this_policy
        self._P_this_policy = np.zeros((state_size,state_size), dtype=np.float32)
        self._R_this_policy = np.zeros((state_size,state_size), dtype=np.float32)

    def run(self, method='policy-iteration', tol=1e-8):
        '''
        FUNC: run DP to obtain optimal policy and corresponding value function
        Argument:
            method: which DP to solve the planning problem, 'policy-iteration' or 'value-iteration'
            tol: tolerance of policy evaluation, only used in method 'policy iteration'
        '''
        print('Start running DP using method {}'.format(method))
        err = 1e5 # just a random large number for initial error
        n_iters = 0
        while(err!=0):
            # run one step
            if method=='policy-iteration':
                new_V, new_policy = self._policy_iteration(tol)
            elif method=='value-iteration':
                new_V, new_policy = self._value_iteration()
            else:
                raise NameError('method in JackCar.run(*) should be \'policy-iteration\' or \'value-iteration\'')
            # check new policy is always better than or equal to the old policy
            tmp = new_V - self._V
            if tmp[tmp<0].any() and n_iters!=0: # if n_iters==0, we still have no valid V to be compared with
                print('Value funtion degenerates, meaning that there must be something wrong in DP')
                break
            # compute error
            err = np.sum(np.absolute(new_policy-self._policy))
            print('policy difference = {}'.format(err))
            # update policy and value function
            self._policy = new_policy
            self._V = new_V
            n_iters += 1

        print('End!! Current policy should be the optimal one')

    def visualize_policy(self):
        '''
        FUNC: visualize policy
        '''
        policy = self.policy
        
        print('')
        print('Optimal policy:')
        for i in range(self._max_cars_A+2):
            if i==0:
                line = ' 0 '
            elif i==int((self._max_cars_A+2)/2):
                line = ' A '
            else:
                line = '   '

            if i==self._max_cars_A+1:
                line += 'V'
            elif i!=0:
                line += '|'

            for j in range(self._max_cars_B):
                if i==0:
                    if j==int((self._max_cars_B)/2):
                        line += ' B '
                    else:
                        line += '   '
                elif i==1:
                    if j==self._max_cars_B-1:
                        line += '-->{:3}'.format(self._max_cars_B)
                    else:
                        line += '---'
                else:
                    line += '{:3}'.format(policy[i-2][j])
            print(line)
        print(' {:3}'.format(self._max_cars_A))
        
        print('+: move cars from A to B')
        print('-: move cars from B to A')

        print('')

    def save_full_env_dynamics(self, fname):
        '''
        FUNC: save full environment dynamics and problem setting to 3 files,
              fname_conf.npy, fname_P.npy, and fname_R.npy
        Argument:
            fname: prefix filename to be saved as
        '''
        # save configuration file
        conf = {
            'max_move': self._max_move,
            'max_cars_A': self._max_cars_A,
            'max_cars_B': self._max_cars_B,
            'rent_price': self._rent_price,
            'move_cost': self._move_cost,
            'gamma': self._gamma
        }
        conf_fname = fname + '_conf.npy'
        np.save(conf_fname, conf)
        # save P_full
        P_fname = fname + '_P.npy'
        np.save(P_fname, self._P_full)
        # save R_full
        R_fname = fname + '_R.npy'
        np.save(R_fname, self._R_full)

        print('Files saved to {}, {}, and {}'.format(conf_fname, P_fname, R_fname))

    def _policy_iteration(self, tol):
        '''
        FUNC: take one step using policy iteration
        '''
        raise NotImplementedError
        '''
        # obtain P_this_policy and R_this_policy
        self._get_PR_this_policy()

        state_size = (self._max_cars_A+1) * (self._max_cars_B+1)

        ### policy evaluation ###
        err = 1e5
        V = self._V
        while(err>tol):
            # compute new value function
            V_rep = np.tile(V, (state_size, 1))
            new_V = np.sum(self._P_this_policy * (self._R_this_policy + self._gamma*V_rep), axis=1)
            # compute SSE (sum of square error)
            err = np.sum(np.square(new_V - V))
            # update V
            V = new_V

        ### policy improvement ###
        # compute value function of trying different actions in current timestep
        score = np.zeros((self._n_acts, state_size))
        V_rep = np.tile(V, (state_size, 1))
        for act_idx in range(self._n_acts):
            score[act_idx] = np.sum(self._P_full[act_idx] * (self._R_full[act_idx] + self._gamma*V_rep), axis=1)
        # obtain new policy
        new_policy = np.argmax(score, axis=0)
        new_policy -= self._max_move # from 0~n_acts to -max_move~max_move

        return new_V, new_policy
        '''

    def _value_iteration(self):
        '''
        FUNC: take one step using value iteration
        '''
        raise NotImplementedError
        '''
        state_size = (self._max_cars_A+1) * (self._max_cars_B+1)
        V = self._V.copy()
        policy = np.zeros_like(self._policy)
        score = np.zeros((self._n_acts))
        for s in range(state_size):
            for act_idx in range(self._n_acts):
                score[act_idx] = np.sum(self._P_full[act_idx,s] * (self._R_full[act_idx,s] + self._gamma*V))
            policy[s] = np.argmax(score) - self._max_move
            V[s] = np.max(score)
        
        return V, policy
        '''

    def _get_PR_this_policy(self):
        '''
        FUNC: get transition matrix and expected return according to current policy
        '''
        # from -max_move~max_move to 0~n_acts
        act_idx = self._policy + self._max_move
        # obtain P and R following current policy
        AB_size = (self._max_cars_A+1) * (self._max_cars_B+1)
        for i in range(AB_size):
            for j in range(AB_size):
                self._P_this_policy[i,j] = self._P_full[act_idx[i], i, j]
                self._R_this_policy[i,j] = self._R_full[act_idx[i], i, j]

    def _build_env_dynamics(self):
        '''
        FUNC: build full environment dynamics, i.e. transition matrix and return matrix
        '''
        print('Forming full environment dynamics, P_full and R_full')
        tic1 = time.time()
        AB_size = (self._max_cars_A+1) * (self._max_cars_B+1)
        tmp = self._max_cars_B + 1
        for i in range(AB_size):
            n_A = int(i / tmp)
            n_B = i % tmp
            for j in range(AB_size):
                n_A_next = int(j / tmp)
                n_B_next = j % tmp
                for act in range(self._n_acts):
                    p, r = self._compute_pr(n_A, n_B, n_A_next, n_B_next, act)
                    self._P_full[act,i,j] = p
                    self._R_full[act,i,j] = r
        toc1 = time.time()
        print('End! Elapsed time = {}'.format(toc1-tic1))

    def _compute_pr(self, n_A, n_B, n_A_next, n_B_next, act):
        '''
        FUNC: compute transition probability and expected return given triplets (this_state,next_state,action)
        '''
        raise NotImplementedError
        '''
        # from 0~n_acts to -max_move~max_move, +: A-->B, -: B-->A
        act = act - self._max_move
        # cars moved from one location to the other cannot > the number of cars at the location
        if (act>0 and act>n_A) or (act<0 and -act>n_B):
            return 0, 0
        # maximum number of cars which can be rented
        A_max_rent = n_A - act
        B_max_rent = n_B + act
        # max_rent cannot surpass maximum storage
        if A_max_rent>self._max_cars_A or B_max_rent>self._max_cars_B:
            return 0, 0
        # difference of cars number in current state(today) and next state(tommorrow)
        A_diff = n_A_next - A_max_rent
        B_diff = n_B_next - B_max_rent
        ### go through all possibilties from cars number as 'n_X' to 'n_X_next' with 'act' done overnight 
        # in location A
        r_A = p_A = 0
        for A_rent in range(A_max_rent,-1,-1): # loop from max_rent to 0 --> work with if-break
            A_return = A_rent + A_diff
            # number of cars returned cannot be negative
            if A_return<0:
                break
            tmp = env.A_return_prob(A_return) * env.A_rent_prob(A_rent)
            r_A += (A_rent*self._rent_price) * tmp
            p_A += tmp
        # in location B
        r_B = p_B = 0
        for B_rent in range(B_max_rent,-1,-1):
            B_return = B_rent + B_diff
            # number of cars returned cannot be negative
            if B_return<0:
                break
            tmp = env.B_return_prob(B_return) * env.B_rent_prob(B_rent)
            r_B += (B_rent*self._rent_price) * tmp
            p_B += tmp
        # compute total expected reward and transition probability
        r = r_A + r_B - np.absolute(act)*self._move_cost
        p = p_A * p_B

        return p, r
        '''
    # property decorator
    @property
    def policy(self):
        n_A = self._max_cars_A + 1
        n_B = self._max_cars_B + 1
        return np.reshape(self._policy, (n_A,n_B))
    @property
    def V(self):
        return self._V
    @property
    def max_move(self):
        return self._max_move
    @property
    def max_cars_A(self):
        return self._max_cars_A
    @property
    def max_cars_B(self):
        return self._max_cars_B
    @property
    def rent_price(self):
        return self._rent_price
    @property
    def move_cost(self):
        return self._move_cost
    @property
    def gamma(self):
        return self._gamma

