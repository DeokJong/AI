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
        for _ in range(self.iterations):
            values_copy = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                max_value = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_value:
                        max_value = q_value
                values_copy[state] = max_value
            self.values = values_copy



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        q_value = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            q_value += prob * (reward + self.discount * self.values[nextState])
        return q_value


    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None
        
        possible_actions = self.mdp.getPossibleActions(state)
        if not possible_actions:
            return None
        
        best_action = None
        max_value = float('-inf')
        
        for action in possible_actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_value:
                max_value = q_value
                best_action = action
                
        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
            # Initialize a priority queue
            priority_queue = util.PriorityQueue()

            # Precompute predecessors of all states
            predecessors = {}
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    for action in self.mdp.getPossibleActions(state):
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                            if prob > 0:
                                if next_state in predecessors:
                                    predecessors[next_state].add(state)
                                else:
                                    predecessors[next_state] = {state}
            
            # Initialize the priority queue with all non-terminal states
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    max_q_value = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                    diff = abs(self.values[state] - max_q_value)
                    priority_queue.push(state, -diff)

            # Perform updates
            for _ in range(self.iterations):
                if priority_queue.isEmpty():
                    break
                state = priority_queue.pop()

                # Update the value of the state
                if not self.mdp.isTerminal(state):
                    max_q_value = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                    self.values[state] = max_q_value

                # Update the values of predecessors
                for pred in predecessors.get(state, []):
                    if not self.mdp.isTerminal(pred):
                        max_q_value = max([self.computeQValueFromValues(pred, action) for action in self.mdp.getPossibleActions(pred)])
                        diff = abs(self.values[pred] - max_q_value)
                        if diff > self.theta:
                            priority_queue.update(pred, -diff)

