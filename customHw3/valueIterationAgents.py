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
        """
        computeActionFromValues를 통해서 argmax_a Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')] 까지 구함.
        argmax를 구했으니 Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')]를 이용하는 computeQValueFromValues를 통해 업데이트
        이것을 이용해 V_k+1(s)를 업데이트 시켜줘야함.
        V_k+1(s)는 업데이트 모든 상태에 대해 업데이트 하는것임
        끝나는 지점은? self.iterations만큼 돌리고 끝내야함.
        """
        
        def showMDPStruct(mdp : mdp.MarkovDecisionProcess):
            """MDP 구조 알아보기
            Args:
                mdp (mdp.MarkovDecisionProcess): 입력값은 mdp
            간단하게 MDP의 구조를 보여주는 함수
            """
            states = mdp.getStates()
            stateInformations = []

            for state in states:
                possibleActions = mdp.getPossibleActions(state)
                for action in possibleActions:
                    transitions = mdp.getTransitionStatesAndProbs(state, action)
                    for nextState, prob in transitions:
                        reward = mdp.getReward(state, action, nextState)
                        stateInformations.append((state, action, nextState, prob, reward))
                        
            for stateInformation in stateInformations:
                print("This is State Information:\nstate, action, nextState, probability, reward\n", stateInformation)
            

        # Write value iteration code here
        # V_k+1(s) ← max_a Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')]
        
        for _ in range(self.iterations):
            lastIterationValues = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                
                max_a_q_value = self.computeQValueFromValues(state, self.computeActionFromValues(state))
                        
                lastIterationValues[state] = max_a_q_value
                
            # value 딕셔너리 업데이트
            self.values = lastIterationValues
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          입력된 State에 입력된 action을 취했을때 나오는 Q value를 리턴함 
          Q value = Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')]
          self.values에 의해 제공된 가치 함수에 따라 (state,action) 쌍의 Q-value를 리턴한다.
          getValue 함수는 state에 대한 value값을 저장하는 딕셔너리에서 입력값과 일치하는 value를 리턴
        """

        # V_k+1(s) ← max_a Σ_s' T(s, a, s')[R(s, a, s') + γV_k(s')]
        
        # Q(s) = Σ_s' probability * (reward + gamma * value(s'))
        q_vlaue_per_nextState = []
        for nextState,probability in self.mdp.getTransitionStatesAndProbs(state,action):
            # per s' Q = probability * (reward + gamma(self.discount) * value(s'))
            q_vlaue_per_nextState.append(probability*(self.mdp.getReward(state,action,nextState) + self.discount * self.values[nextState]))

        # Σ_s'
        Q_value = sum(q_vlaue_per_nextState)
        
        return Q_value

    def computeActionFromValues(self, state):
        """
        state가 주워졌을때 해당 state에서 가능한 action들을 취했을때
        그중 가장 큰 Q값을 만드는 action을 리턴함
        argmax에 해당하는 함수임
        """
        
        # 가능한 action
        possibleActions = self.mdp.getPossibleActions(state)
        
        # action이 없으면 None 리턴
        if not possibleActions or self.mdp.isTerminal(state):
            return None
        
        bestAction = None
        bestValue = float('-inf')
        
        # best action 고르기
        for action in possibleActions:
            currentQValue = self.computeQValueFromValues(state,action)
            if bestValue < currentQValue:
                bestValue = currentQValue
                bestAction = action
        
        # 가장 큰 Q값을 만드는 action 리턴
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
            # 모든 state에 대해 predecessros
            predecessors = {}
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    for bestAction in self.mdp.getPossibleActions(state):
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, bestAction):
                            if prob > 0:
                                predecessors.setdefault(next_state, set()).add(state)
            
            # 우선순위 큐 선언
            pq = util.PriorityQueue()

            # 각 non-terminal state에 대해 state를 -diff 우선순위로 push
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    bestAction = self.computeActionFromValues(state)
                    maxQValue = self.computeQValueFromValues(state, bestAction)
                    diff = abs(self.values[state] - maxQValue)
                    pq.push(state, -diff)

            # iteration만큼 돌리는데,
            for _ in range(self.iterations):
                # pq가 비어있으면 break
                if pq.isEmpty():
                    break

                # 비어있지 않으면 pop하고, 해당 state를 
                state = pq.pop()
                
                # state가 terminal이 아니면, slef.values 업데이트
                if not self.mdp.isTerminal(state):
                    bestAction = self.computeActionFromValues(state)
                    maxQValue = self.computeQValueFromValues(state, bestAction)
                    self.values[state] = maxQValue

                # 업데이트가 끝나면 이 state가 목적지인 predecessor들을 업데이트
                for predecessor in predecessors.get(state, []):
                    if not self.mdp.isTerminal(predecessor):
                        bestAction = self.computeActionFromValues(predecessor)
                        maxQValue = self.computeQValueFromValues(predecessor, bestAction)
                        diff = abs(self.values[predecessor] - maxQValue)
                        # 단 diff가 theta보다 크면 pq에 업데이트
                        if diff > self.theta:
                            pq.update(predecessor, -diff)
