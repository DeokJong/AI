# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          상태와 액션를 키로 사용하여
          Q값 딕셔너리에서 그에 해당하는 Q값을 리턴함
        """
        return self.qValues[(state,action)]


    def computeValueFromQValues(self, state):
        """
          상태가 주워졌을때 그 상태에서 가장 큰 Q값을 리턴함
          만일 그 상태에서 액션을 취할 수 없을때 0값을 리턴함
        """
        actions = self.getLegalActions(state)
        
        if not actions :
          return 0.0
        
        max_q_value = max(self.getQValue(state,action) for action in actions)
        
        return max_q_value

    def computeActionFromQValues(self, state):
        """
          해당 상태에서 Q값이 최대로 하게 하는 action을 얻음
          만일 그 상태에서 액션을 취할 수 없을때 None을 리턴함
          만일 최대 Q값을 가지는 액션이 여러개일때 랜덤으로 선택함
          argmax_a for Q
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None
        
        bestValue = self.computeValueFromQValues(state)
        bestActions = [action for action in actions if self.getQValue(state, action) == bestValue]

        return random.choice(bestActions)

    def getAction(self, state):
        """
          입실론 확률로는 랜덤 액션을 선택하고, 그렇지 않으면
          computeActionFromQValues를 통해 최적의 액션을 선택함
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if not legalActions:
            return None
        
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          Q값을 업데이트함

          Q(s,a) = (1-alpha) * Q(s,a) + alpha * sample)
          sample = R + gamma * max_a Q(s',a)
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state,action)] = (1-self.alpha) * self.getQValue(state,action) + self.alpha * sample
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
         가중치와 feature을 사용하여 Q값을 계산함

         feature는 state와 action을 사용하여 계산함
         가중치는 getWeights를 통해 얻음
         
         둘은 모두 벡터이므로 내적을 통해 Q값을 계산함
        """

        features = self.featExtractor.getFeatures(state, action)
        current_weights = self.getWeights()

        q_value = sum(current_weights[iter] * features[iter] for iter in features)

        return q_value

    def update(self, state, action, nextState, reward):
        """
         가중치를 업데이트함
         w_i = w_i + alpha * difference * f_i(s,a)
         diff = sample - Q(s,a)
         sample = R + gamma * max_a Q(s',a)
        """
        
        lastWeight = self.getWeights()
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        difference = sample - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)

        for feature in features:
            self.weights[feature] = lastWeight[feature] + self.alpha * difference * features[feature]
        


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print('Weights:', self.getWeights())
            "*** YOUR CODE HERE ***"
            pass
