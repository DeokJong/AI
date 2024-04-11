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
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
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
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
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
    def TSP(startState,goalStates,getSuccessors):
        import itertools
        edgeDic={}
        preStatesList = itertools.permutations(goalStates)
        
        statesList = [(startState,) + perm for perm in preStatesList]
        print(statesList)
        for states in statesList:
            edgeDic[states] = simpleBFS(states[0],states[1],getSuccessors)+simpleBFS(states[1],states[2],getSuccessors)+simpleBFS(states[2],states[3],getSuccessors)+simpleBFS(states[3],states[4],getSuccessors)
        
        bestCost=float('inf')
        bestEdge=[]
        for edge in edgeDic:
            currCost = len(edgeDic[edge])
            if bestCost > currCost:
                bestCost=currCost
                bestEdge=edgeDic[edge]
        
        return bestEdge
        
    try:
        goals = problem.corners
        print("This is corner Problem")
        return TSP(problem.getStartState(),goals,problem.getSuccessors)
    except:
        print("NOT corner Problem!")
        return normalBFS(problem.getStartState(),problem.isGoalState,problem.getSuccessors)
        
def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
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
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
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
