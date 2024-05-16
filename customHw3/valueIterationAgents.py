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
    def __init__(self, mdp : mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
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
        # class StateStruct:
        #     def __init__(self, mdp:mdp.MarkovDecisionProcess):
        #         self.mdp = mdp
        #         self.states = self.mdp.getStates()
        #         self.startState = self.mdp.getStartState()
        #         self.stateInformations = []

        #         for state in self.states:
        #             possibleActions = self.mdp.getPossibleActions(state)
        #             for action in possibleActions:
        #                 transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        #                 for nextState, prob in transitions:
        #                     reward = self.mdp.getReward(state, action, nextState)
        #                     self.stateInformations.append((state, action, nextState, prob, reward))
                            
        #         self.showMDPInformation()

        #     def showMDPInformation(self):
        #         for stateInformation in self.stateInformations:
        #             print("This is State Information:\nstate, action, nextState, probability, reward\n", stateInformation)
    
        # currentStateStruct = StateStruct(self.mdp)

        # Write value iteration code here
        # V_k+1(s) ← max_a Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')]
        """
        computeActionFromValues를 통해서 argmax_a Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')] 까지 구함.
        argmax를 구했으니 Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')]를 이용하는 computeQValueFromValues를 통해 업데이트
        이것을 이용해 V_k+1(s)를 업데이트 시켜줘야함.
        V_k+1(s)는 업데이트 모든 상태에 대해 업데이트 하는것임
        끝나는 지점은? self.iterations만큼 돌리고 끝내야함.
        """
        mdpStates = self.mdp.getStates()
        tempValuesForState = self.values.copy()
        for _ in range(self.iterations):
            for mdpState in mdpStates:
                if self.mdp.isTerminal(mdpState):
                    continue
                
                bestActionForThisMdpState = self.computeActionFromValues(mdpState)
                tempValuesForState[mdpState] = self.computeQValueFromValues(mdpState,bestActionForThisMdpState)
                
            self.values = tempValuesForState
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          
          self.values에 의해 제공된 가치 함수에 따라 (state,action) 쌍의 Q-value를 리턴한다.
          getValue 함수는 state에 대한 value값을 저장하는 딕셔너리에서 입력값과 일치하는 value를 리턴
        """

        # V_k+1(s) ← max_a Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')]
        
        # Q(s) = Σ_s' probability * (reward + gamma * value(s'))
        q_vlaue_per_nextState = []
        for nextState,probability in self.mdp.getTransitionStatesAndProbs(state,action):
            reward = self.mdp.getReward(state,action,nextState)
            # per s' Q = probability * (reward + gamma(self.discount) * value(s'))
            q_vlaue_per_nextState.append(probability*(reward + self.discount * self.values[nextState]))

        # Σ_s'
        Q_value = sum(q_vlaue_per_nextState)
        
        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
          
          self.values에 의해 제공된 가치 함수에 따라 최적의 action을 계산해야 한다.
          computeQValueFromValues 통해서 state당 Q값을 찾았음.
          이제 computeActionFromValues를 통해서 max가 되게하는 a값 즉 bestAction을 찾아서 리턴.
        """
        possibleActions = self.mdp.getPossibleActions(state)
        
        if not possibleActions or self.mdp.isTerminal(state):
            return None
        
        bestAction = None
        bestValue = float('-inf')
        
        for action in possibleActions:
            currentQValue = self.computeQValueFromValues(state,action)
            if bestValue < currentQValue:
                bestValue = currentQValue
                bestAction = action
        
        return bestAction
        

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
        "*** YOUR CODE HERE ***"

