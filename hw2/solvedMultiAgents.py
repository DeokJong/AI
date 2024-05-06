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
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # 동작 후의 상태
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # 음식과의 거리 점수
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            minFoodDistance = min(foodDistances)
        else:
            minFoodDistance = 0

        # 유령과의 거리 점수
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances) if ghostDistances else 10
        ghostScore = -10 if minGhostDistance < 2 else 0  # 유령과 2칸 이내일 경우 큰 페널티

        # 정지 동작 피하기
        if action == Directions.STOP:
            stopPenalty = -20
        else:
            stopPenalty = 0

        # 최종 점수 계산
        score = successorGameState.getScore() + 1.0 / (minFoodDistance + 1) + ghostScore + stopPenalty
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth if nextAgent != 0 else depth + 1

            if agentIndex == 0:  # Maximizing for Pacman
                bestScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = minimax(nextAgent, nextDepth, successor)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                return bestScore, bestAction
            else:  # Minimizing for ghosts
                worstScore = float('inf')
                worstAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = minimax(nextAgent, nextDepth, successor)
                    if score < worstScore:
                        worstScore = score
                        worstAction = action
                return worstScore, worstAction

        # Call the minimax function starting from Pacman (agentIndex = 0) and depth 0
        _, action = minimax(0, 0, gameState)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        알파-베타 프루닝을 사용하여 미니맥스 액션을 반환합니다.
        """

        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            # 종료 조건: 게임이 끝났거나 최대 깊이에 도달한 경우
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth if nextAgent > 0 else depth + 1

            # 유령의 턴 (최소화 에이전트)
            if agentIndex > 0:
                value = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(nextAgent, nextDepth, successor, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

            # 팩맨의 턴 (최대화 에이전트)
            else:
                value = float('-inf')
                bestAction = Directions.STOP
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    currentValue = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                    if currentValue > value:
                        value, bestAction = currentValue, action
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                if depth == 0:
                    return bestAction
                else:
                    return value

        # 알파-베타 탐색 초기 호출
        return alphaBeta(0, 0, gameState, float('-inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
        게임의 현재 상태에서 Expectimax 알고리즘을 사용하여 최적의 행동을 반환합니다.
        """

        def expectimax(agentIndex, depth, gameState):
            # 종료 조건: 게임이 끝났거나 최대 깊이에 도달한 경우
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth if nextAgent > 0 else depth + 1

            actions = gameState.getLegalActions(agentIndex)

            # 유령의 턴 (기대값 계산)
            if agentIndex > 0:
                results = []
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    results.append(expectimax(nextAgent, nextDepth, successor))
                return sum(results) / len(results)  # 기대값 계산

            # 팩맨의 턴 (최대값 계산)
            else:
                results = []
                bestAction = Directions.STOP
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    currentValue = expectimax(nextAgent, nextDepth, successor)
                    results.append((currentValue, action))
                bestValue, bestAction = max(results)
                if depth == 0:
                    return bestAction  # 최상위 호출에서는 최적의 행동 반환
                else:
                    return bestValue  # 재귀 호출에서는 최적의 값을 반환

        # Expectimax 탐색의 초기 호출
        return expectimax(0, 0, gameState)


def betterEvaluationFunction(currentGameState: GameState):
    """
    DESCRIPTION: This evaluation function aims to maximize Pacman's score by considering:
    - The distance to the nearest and average food.
    - The number of remaining food items.
    - The distance to the nearest active ghost and the nearest scared ghost.
    - The location of power pellets.
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    powerPellets = currentGameState.getCapsules()
    
    # Food related evaluation
    foodList = newFood.asList()
    foodDistances = [manhattanDistance(newPos, food) for food in foodList]
    if foodDistances:
        minFoodDistance = min(foodDistances)
        avgFoodDistance = sum(foodDistances) / len(foodDistances)
    else:
        minFoodDistance = avgFoodDistance = 0

    # Ghost related evaluation
    ghostDistances = []
    scaredGhostDistances = []
    for ghostState in newGhostStates:
        distance = manhattanDistance(newPos, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            scaredGhostDistances.append(distance)
        else:
            ghostDistances.append(distance)

    if ghostDistances:
        minGhostDistance = min(ghostDistances)
    else:
        minGhostDistance = float('inf')

    if scaredGhostDistances:
        minScaredGhostDistance = min(scaredGhostDistances)
    else:
        minScaredGhostDistance = float('inf')

    # Power pellets evaluation
    pelletDistances = [manhattanDistance(newPos, pellet) for pellet in powerPellets]
    minPelletDistance = min(pelletDistances) if pelletDistances else 0

    # Final score computation
    score = currentGameState.getScore()
    score -= 1.5 * avgFoodDistance  # prioritize average food distance
    score += 2 / (1 + minGhostDistance)  # high penalty for close active ghosts
    score -= 2 * (1 / (1 + minScaredGhostDistance)) if minScaredGhostDistance != float('inf') else 100
    score -= 4 * minPelletDistance  # encourage power pellet consumption
    score -= 100 * len(foodList)  # penalty for more remaining food

    return score

# Abbreviation
better = betterEvaluationFunction
