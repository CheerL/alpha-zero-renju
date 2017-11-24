"""Monte Carlo Tree Search, as described in Silver et al 2015.
This is a "pure" implementation of the AlphaGo MCTS algorithm in that it is not specific to the
game of Go; everything in this file is implemented generically with respect to some state, actions,
policy function, and value function.
"""
from copy import deepcopy

import numpy as np
from scipy.stats import dirichlet


class MCTNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}      # a map from action
        self.P = prior_prob     # prior prob
        self.N = 0              # visit time
        self.W = 0              # total action value
        self.Q = 0              # mean action value

    def get_Q_plus_U(self):
        '''Q + U'''
        c_puct = 5
        U = c_puct * self.P * (self.parent.N ** 0.5) / (1 + self.N)
        return self.Q + U

    def expand(self, action_probs):
        """Expand tree by creating new children.
        Arguments:
        action_priors -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.
        Returns:
        None
        """
        for action, prob in action_probs.items():
            self.children[action] = MCTNode(self, prob)

    def select(self):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        return max(self.children.items(), key=lambda action_node: action_node[1].Q_plus_U())

    def backup(self, value):
        """Update node values from leaf evaluation.
        Arguments:
        value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        Returns:
        None
        """
        if self.parent:
            self.parent.backup(value)
        
        # Update u, the prior weighted by an exploration hyperparameter c_puct and the number of
        # visits. Note that u is not normalized to be a distribution.
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def is_leaf(self):
        return self.children == {}

class MCT(object):
    """A simple (and slow) single-threaded implementation of Monte Carlo Tree Search.
    Search works by exploring moves randomly according to the given policy up to a certain
    depth, which is relatively small given the search space. "Leaves" at this depth are assigned a
    value comprising a weighted combination of (1) the value function evaluated at that leaf, and
    (2) the result of finishing the game from that leaf according to the 'rollout' policy. The
    probability of revisiting a node changes over the course of the many playouts according to its
    estimated value. Ultimately the most visited node is returned as the next action, not the most
    valued node.
    The term "playout" refers to a single search from the root, whereas "rollout" refers to the
    fast evaluation from leaf nodes to the end of the game.
    """

    def __init__(self, evaluate_fn, size, max_evaluate_time=1600, tau=1):
        """Arguments:
        value_fn -- a function that takes in a state and ouputs a score in [-1, 1], i.e. the
            expected value of the end game score from the current player's perspective.
        policy_fn -- a function that takes in a state and outputs a list of (action, probability)
            tuples for the current player.
        rollout_policy_fn -- a coarse, fast version of policy_fn used in the rollout phase.
        lmbda -- controls the relative weight of the value network and fast rollout policy result
            in determining the value of a leaf node. lmbda must be in [0, 1], where 0 means use only
            the value network and 1 means use only the result from the rollout.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more, and
            should be used only in conjunction with a large value for n_playout.
        """
        self.root = MCTNode(None, 1.0)
        self.full_size = size ** 2
        self.evaluate_fn = evaluate_fn                  # network evaluate function
        self.max_evaluate_time = max_evaluate_time      # max evaluate time
        self.tau = tau                                  # temperature para
                                                        # round < 30    : 1
                                                        # round >= 30   : 0.01
        self.dirichlet_noise_distribute = None

    def play(self, state):
        """Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        leaf_depth -- after this many moves, leaves are evaluated.
        Returns:
        None
        """
        evaluate_time = 0
        while evaluate_time < self.max_evaluate_time:
            node = self.root
            temp_state = deepcopy(state)
            # go down to leaf node
            while not node.is_leaf():
                action, node = node.select()
                temp_state.move(action)
            # come a leaf node
            action_probs, value = self.evaluate(temp_state)
            node.expand(action_probs)
            node.backup(value)
            evaluate_time += 1
            del temp_state

    def evaluate(self, state):
        """Use the rollout policy to play until the end of the game, returning +1 if the current
        player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        # net work evaluate + Dirichlet noise
        if not self.dirichlet_noise_distribute:
            alpha = np.ones(self.full_size) * 0.03
            self.dirichlet_noise_distribute = dirichlet(alpha)

        noise_rate = 0.25
        noise = self.dirichlet_noise_distribute.rvs()[0]
        raise NotImplementedError()

    def get_move_probability(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        Arguments:
        state -- the current state, including both game state and the current player.
        Returns:
        the selected action
        """
        temperature_para = 1 / self.tau
        move_probability = np.array(
            [child ** temperature_para for child in self.root.children.values()],
            np.float
            )
        return move_probability / move_probability.sum()

    def update(self, move, oppo_move):
        """Step forward in the tree, keeping everything we already know about the subtree, assuming
        that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        def _update(last_move):
            if self.root.children and last_move in self.root.children:
                self.root = self.root.children[move]
            else:
                self.root = MCTNode(None, 1.0)

        _update(move)
        _update(oppo_move)

    def get_move(self, probability):
        move_index = np.random.choice(np.arange(self.full_size), p=probability)
        return move_index
