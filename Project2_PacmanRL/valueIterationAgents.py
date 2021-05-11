# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import copy

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        new_values = copy.deepcopy(self.values)
        for i in range(self.iterations):
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    new_values[state] = max([self.computeQValueFromValues(state, action) \
                                              for action in self.mdp.getPossibleActions(state)])
            self.values = copy.deepcopy(new_values)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # call this to apply bellman update
        if self.mdp.isTerminal(state):
            return self.values[state]
        else:
            return sum([prob*(self.mdp.getReward(state, action, next_state) + self.discount*self.values[next_state]) \
                         for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # be wary of edge cases
        if self.mdp.isTerminal(state):
            return None
        else:
            best_action = ''
            best_val = -float('inf')
            for act in self.mdp.getPossibleActions(state):
                act_val = 0
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, act):
                    act_val += prob*self.getValue(next_state)
                if act_val > best_val:
                    best_action = act
                    best_val = act_val
            return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        num_states = len(self.mdp.getStates())
        for i in range(self.iterations):
            state = self.mdp.getStates()[i%num_states]
            if state == 'TERMINAL_STATE':
                continue
            else:
                self.values[state] = max([self.computeQValueFromValues(state, action) \
                                         for action in self.mdp.getPossibleActions(state)])

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        preds = {s: set() for s in self.mdp.getStates()}
        for pred in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(pred):
                for succ, p in self.mdp.getTransitionStatesAndProbs(pred, action):
                    if p > 0:
                        preds[succ].add(pred)

        queue = util.PriorityQueue()
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                maxQ = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
                diff = abs(self.values[s] - maxQ)
                queue.update(s, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            for p in preds[s]:
                if not self.mdp.isTerminal(p):
                    maxQ = max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)])
                    diff = abs(maxQ - self.values[p])
                    if diff > self.theta:
                        queue.update(p, -diff)