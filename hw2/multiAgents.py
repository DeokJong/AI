# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):

    def getAction(self, gameState: GameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        
        minFoodDistance = min([manhattanDistance(newPos, food) for food in newFood.asList()]) if newFood.asList() else -1

        score = successorGameState.getScore() + 1/minFoodDistance 

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState):
        """
        V(s) = max_a V(s'), where s' = result(s,a)
        s에서 가능한 a들을 모두 취하고 그중에서 가장 높은값 선택
        
        function minimax_decision( state )
            return argmax a in state.actions value( state.result(a) )
            
        function value( state )
            if state.is_leaf
                return state.value
            if state.player is MAX
                return max a in state.actions value( state.result(a) )
            if state.player is MIN
                return min a in state.actions value( state.result(a) )
                
        일반적으로 하기 위해선 argmax를 주로 구현
        """
        
        def getNextAgentIndex(currentAgentIndex):
            totalAgentNumber = gameState.getNumAgents()
            nextAgentIndex = (currentAgentIndex+1) % totalAgentNumber
            return nextAgentIndex
        
        # agentIndex = state.player
        # depth = state.depth
        def getValue(currentAgentIndex, currentDepth, gameState : GameState):
            
            # if state.is_leaf
            if  gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                # return state.value and None action
                return self.evaluationFunction(gameState),None
            
            # init s' state
            nextAgentIndex = getNextAgentIndex(currentAgentIndex)
            nextDepth = currentDepth if nextAgentIndex !=0 else currentDepth+1
            
            # if state.player is MAX
            if currentAgentIndex == 0:
                bestScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(currentAgentIndex):
                    # aftetState == s'
                    nextGameState = gameState.generateSuccessor(currentAgentIndex,action)
                    minScore, _ = getValue(nextAgentIndex,nextDepth,nextGameState)
                    # a' argmax_a V(s')
                    if minScore > bestScore:
                        bestScore = minScore
                        bestAction = action
                return bestScore,bestAction
            
            # if state.player == MIN
            else:
                bestScore = float('inf')
                bestAction = None
                for action in gameState.getLegalActions(currentAgentIndex):
                    nextGameState = gameState.generateSuccessor(currentAgentIndex,action)
                    maxScore,_ = getValue(nextAgentIndex,nextDepth,nextGameState)
                    if maxScore < bestScore:
                        bestScore = maxScore
                        bestAction = action
                return bestScore,bestAction
        
        # a = argmax_a(s) 
        _, bestActions = getValue(0,0,gameState)
        return bestActions

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        def getNextAgentIndex(currentAgentIndex):
            totalAgentNumber = gameState.getNumAgents()
            nextAgentIndex = (currentAgentIndex+1) % totalAgentNumber
            return nextAgentIndex
        
        print("start get action function")
        def getAlphaBeta(currentAgentIndex, currentDepth, gameState : GameState, globalAlpha, globalBeta):
            # if gameEnd or max Depth return curretn evaluationFuction value
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                print("end depth\n")
                return self.evaluationFunction(gameState)

            # assign next agent
            nextAgent = getNextAgentIndex(currentAgentIndex)
            nextDepth = currentDepth if nextAgent > 0 else currentDepth + 1

            # Max agent
            # find alpha
            # if alpha > beta then pruning
            if currentAgentIndex == 0:
                print("Get alpha")
                # init current alpha
                alpha = float('-inf')
                bestAction = Directions.STOP
                
                # max_a and argmax_a
                for action in gameState.getLegalActions(currentAgentIndex):
                    nextGameState = gameState.generateSuccessor(currentAgentIndex, action)
                    beta = getAlphaBeta(nextAgent, nextDepth, nextGameState, globalAlpha, globalBeta)
                    if beta > alpha:
                        # update alpha
                        alpha, bestAction = beta, action
                    if alpha > globalBeta:
                        # if alpha > beta then pruning
                        print("Pruning at beta")
                        return alpha
                    # init define current alpha
                    globalAlpha = max(globalAlpha, alpha)
                return bestAction if currentDepth == 0 else alpha

                

            # Min agent
            # find beta
            # if alpha > beta then pruning
            else:
                print("Get beta")
                # init current beta
                alpha = float('inf')
                
                # min_a and argmin_a
                for action in gameState.getLegalActions(currentAgentIndex):
                    nextGameState = gameState.generateSuccessor(currentAgentIndex, action)
                    alpha = min(alpha, getAlphaBeta(nextAgent, nextDepth, nextGameState, globalAlpha, globalBeta))
                    if alpha < globalAlpha:
                        print("Pruning at alpha")
                        return alpha
                    # init define current beta
                    globalBeta = min(globalBeta, alpha)
                return alpha
               
        # init call
        # insert root alpha(-inf), root beta(inf)
        return getAlphaBeta(0, 0, gameState, float('-inf'), float('inf'))
                
        

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState):
        
        def getNextAgentIndex(currentAgentIndex):
            totalAgentNumber = gameState.getNumAgents()
            nextAgentIndex = (currentAgentIndex+1) % totalAgentNumber
            return nextAgentIndex
        
        def getExpectiMax(currentAgentIndex,currentDepth,gameState:GameState):
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            
            nextAgentIndex = getNextAgentIndex(currentAgentIndex)
            nextDepth = currentDepth if nextAgentIndex > 0 else currentDepth+1
            
            actions = gameState.getLegalActions(currentAgentIndex)
            
            if currentAgentIndex == 0:
                ableResultValueAndActionList = []
                bestAction = Directions.STOP
                
                for action in actions:
                    nextGameState = gameState.generateSuccessor(currentAgentIndex,action)
                    currentValue = getExpectiMax(nextAgentIndex,nextDepth,nextGameState)
                    ableResultValueAndActionList.append((currentValue,action))
                bestValue, bestAction = max(ableResultValueAndActionList)
                
                if currentDepth == 0:
                    return bestAction
                else:
                    return bestValue
            
            else:
                resultValueList = []
                for action in actions:
                    nextGameState = gameState.generateSuccessor(currentAgentIndex,action)
                    resultValueList.append(getExpectiMax(nextAgentIndex,nextDepth,nextGameState))
                return sum(resultValueList) /len(resultValueList)
                    
            
        return getExpectiMax(0,0,gameState)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    1. Distance to closest ghost
    2. Distance to closest dot
    3. Distance to capsule
    4. Distacne to avg dot
    """
    def getGST(pacmanPos, ghostPositions):
        ghostDistances = []
        for ghostPos in ghostPositions:
            ghostDistances.append(manhattanDistance(pacmanPos,ghostPos))
            
        if ghostDistances:
            return min(ghostDistances)
        else:
            return float('inf')
    
    def getDOT(pacmanPos,foodPositions):
        dotDistances = []
        for foodPos in foodPositions:
            dotDistances.append(manhattanDistance(pacmanPos,foodPos))
            
        minDotDistanceOfFeature = 0
        
        if dotDistances :
            minDotDistanceOfFeature = 1.0/min(dotDistances)
            avgDotDistance = sum(dotDistances)/len(dotDistances)
            return minDotDistanceOfFeature , avgDotDistance
        else:
            return minDotDistanceOfFeature , 0
    
    def getCAP(pacmanPos,capsulePositions):
        capsuleDistances = [manhattanDistance(pacmanPos, capsule) for capsule in capsulePositions]
        minCapsuleDistance = min(capsuleDistances) if capsuleDistances else 0
        return minCapsuleDistance
    
    # assign variable
    pacmanPos = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    foodGrid = currentGameState.getFood()
    foodPositions = foodGrid.asList()
    capsulePositions = currentGameState.getCapsules()
    
    totalScore = 0
    
    # get feature
    currentScore = currentGameState.getScore()
    gst_featrue = getGST(pacmanPos,ghostPositions)
    cap_feature = getCAP(pacmanPos,capsulePositions)
    minDot_feature,avgDot_feature = getDOT(pacmanPos,foodPositions)
    
    # calculate score with feature
    totalScore+=currentScore
    
    totalScore += minDot_feature
    totalScore -= avgDot_feature
    totalScore += 1 / (1+gst_featrue)
    
    return totalScore

# Abbreviation
better = betterEvaluationFunction
