# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    #initialize the explored set to be empty
    visited = set()
    #initialize the frontier as stack
    nodeStack = util.Stack()
    #add initial state of problem th frontier
    nodeStack.push((problem.getStartState(),[]))

    #loop do #if the frontier is empty then return failure
    while not nodeStack.isEmpty():
        #chose a node and remove it from the frontier
        currState, actions = nodeStack.pop()

        #if the node contains a goal state then return the coreesponding solution
        if problem.isGoalState(currState):
            return actions
        
        #add the node state to the explored set
        visited.add(currState)

        #for each resulting child from node 
        successors = problem.getSuccessors(currState)
        for successor in successors:
            nextState, direction, pathCost = successor
            #if the child state is not already in the frontier or explored set then add child to the frontier
            if nextState not in visited:
                nodeStack.push((nextState,actions+[direction]))

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    def normalBFS(startState,isGoalState,getSuccessors):
        print("RUN NORMAL BFS")
        visited = set()
        nodeQueue = util.Queue()
        nodeQueue.push((startState,[]))

        while not nodeQueue.isEmpty():
            #chose a node and remove it from the frontier
            currState, actions = nodeQueue.pop()

            #if the node contains a goal state then return the coreesponding solution
            if isGoalState(currState):
                return actions
            
            #add the node state to the explored set
            if currState not in visited:
                visited.add(currState)

                #for each resulting child from node 
                successors = getSuccessors(currState)
                for successor in successors:
                    nextState, direction, pathCost = successor

                    #if the child state is not already in the frontier or explored set then add child to the frontier
                    if nextState not in visited:
                        nodeQueue.push((nextState,actions+[direction]))

        print("ERROR NORMAL BFS")
        util.raiseNotDefined()
    def simpleBFS(startState,goalState,getSuccessors):
        print("RUN SIMPLE BFS")
        visited = set()
        nodeQueue = util.Queue()
        nodeQueue.push((startState,[]))

        while not nodeQueue.isEmpty():
            currState, actions = nodeQueue.pop()

            if currState==goalState:
                return actions
            
            if currState not in visited:
                visited.add(currState)

                successors = getSuccessors(currState)
                for successor in successors:
                    nextState, direction, pathCost = successor
                    if nextState not in visited:
                        nodeQueue.push((nextState,actions+[direction]))

        print("ERROR SIMPLE BFS")
        util.raiseNotDefined()
    def TSP(problem: SearchProblem):
        initStates = problem.getStartState()
        startState = initStates[0]
        goalStates = initStates[1:]
        
        edgesDic = {
        's_a': simpleBFS(startState,goalStates[0],problem.getSuccessors),
        's_b': simpleBFS(startState,goalStates[1],problem.getSuccessors),
        's_c': simpleBFS(startState,goalStates[2],problem.getSuccessors),
        's_d': simpleBFS(startState,goalStates[3],problem.getSuccessors),
        'a_b': simpleBFS(goalStates[0],goalStates[1],problem.getSuccessors),
        'b_c': simpleBFS(goalStates[1],goalStates[2],problem.getSuccessors),
        'c_d': simpleBFS(goalStates[2],goalStates[3],problem.getSuccessors),
        'a_d': simpleBFS(goalStates[0],goalStates[3],problem.getSuccessors),
        'a_c': simpleBFS(goalStates[0],goalStates[2],problem.getSuccessors),
        'b_d': simpleBFS(goalStates[1],goalStates[3],problem.getSuccessors)
        }
        edgesCostDic = {
        key: len(value) for key, value in edgesDic.items()
        }
        

        bestEdges = getBestEdges(edgesCostDic)
        return adjustEdge(bestEdges,edgesDic)  
    def adjustEdge(edges, edgesDic):
        def getReverseDirection(actions):
            reversedActions=[]
            for action in actions:
                if(action=='North'):
                    reversedActions+=['South']
                elif(action=='South'):
                    reversedActions+=['North']
                elif(action=='East'):
                    reversedActions+=['West']
                elif(action=='West'):
                    reversedActions+=['East']
                else:
                    util.raiseNotDefined()
            return reversedActions

        actions = []
        lastNode = 's'
        print(edges)
        for edge in edges:
            print(edgesDic[edge])
            if edge[0] == lastNode:
                actions.extend(edgesDic[edge])
                print(edgesDic[edge])
                lastNode = edge[2]
            else:
                actions.extend(list(reversed(getReverseDirection(edgesDic[edge]))))
                print(list(reversed(getReverseDirection(edgesDic[edge]))))
                lastNode = edge[0]
            print(actions)
        return actions
    def getBestEdges(edgesCostDic):
        from itertools import permutations

        priority = {'s': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}

        def adjustEdgeByPriority(start, end):
            if priority[start] > priority[end]:
                return f"{end}_{start}"
            else:
                return f"{start}_{end}"

        allPermutations = permutations(['a', 'b', 'c', 'd'])

        allPossibleEdgePathsList = [
            [
                adjustEdgeByPriority('s', perm[0]),
                adjustEdgeByPriority(perm[0], perm[1]),
                adjustEdgeByPriority(perm[1], perm[2]),
                adjustEdgeByPriority(perm[2], perm[3])
            ]
            for perm in allPermutations
        ]

        bestCost = float('inf')
        bestEdges = []

        for path in allPossibleEdgePathsList:
            currCost = 0
            currEdges = []

            for edge in path:
                currCost += edgesCostDic[edge]
                currEdges.append(edge)

            if bestCost > currCost:
                bestCost = currCost
                bestEdges = currEdges
        print(bestCost)
        return bestEdges

    startState = problem.getStartState()
    if(startState == (-1,-1)):
        return TSP(problem)
    else:
        return normalBFS(problem.getStartState(),problem.isGoalState,problem.getSuccessors)

def uniformCostSearch(problem: SearchProblem):
    # initialize the explored set to be empty
    visited = set()
    
    # initialize the frontier as a priority queue using node path_cost as the priority
    nodePriorityQueue = util.PriorityQueue()
    
    # add initial state of problem to frontier with pathCost = 0
    # (state,actions,currCost) == item , (pathCost=currCost+stepCost) == priority 
    nodePriorityQueue.push((problem.getStartState(),[],0),0)
    
    # loop do
    # if the frontier is empty then
    # return failure
    while not nodePriorityQueue.isEmpty():
        # choose a node and remove it from the frontier
        currState,actions,currCost = nodePriorityQueue.pop()
        
        # if the node contains a goal state then
        if problem.isGoalState(currState):
            # return the corresponding solution
            return actions
        
        # add the node state to the explored set
        if currState not in visited:
            visited.add(currState)
            
            # for each resulting child from node
            successors = problem.getSuccessors(currState)
            for successor in successors:
                nextState, action, stepCost = successor
                nextCost = currCost+stepCost
                
                # if the child state is not already in the frontier or explored set then
                if nextState not in visited:
                    # add child to the frontier
                    nodePriorityQueue.update((nextState,actions+[action],nextCost),nextCost)
        
    return[]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    # initialize the explored set to be empty
    visited = set()
    # initialize the frontier as a priority queue using f(n)+g(n)+h(S) as the priority
    nodePriorityQueue = util.PriorityQueue()
    
    # add initial state of problem to frontier with f(S)=g(0)+h(S)
    # (state,actions,currCost) == item , (g(n)+h(S)) == priority, currCost == g(n)
    startState=problem.getStartState()
    goalStates=[]
    if type(startState) == list:
        goalStates = startState[1:]
        startState = startState[0]

    startStateHeuristic = heuristic(startState,problem)
    startStateFn = 0 + startStateHeuristic
    nodePriorityQueue.push((startState,[],0),startStateFn)
    
    # loop do
    # if the frontier is empty then
    # return failure
    while not nodePriorityQueue.isEmpty():
        
        # choose a node and remove it from the frontier
        currState,actions,currCost = nodePriorityQueue.pop()
        
        # if the node contains a goal state then
        if problem.isGoalState(currState):
            # return the corresponding solution
            return actions
        
        # add the node state to the explored set
        if currState not in visited:
            visited.add(currState)
            
            # for each resulting child from node
            successors = problem.getSuccessors(currState)
            for successor in successors:
                nextState, action, stepCost = successor 
                nextStateHeuristic = heuristic(nextState,problem)
                nextCost = stepCost+currCost
                nextStateFn = nextCost + nextStateHeuristic
                
                # if the child state is not already in the frontier or explored set then
                if nextState not in visited:
                    
                    # add child to the frontier
                    nodePriorityQueue.update((nextState,actions+[action],nextCost),nextStateFn)
        
    return[]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


