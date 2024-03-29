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
    #initialize the explored set to be empty
    visited = set()
    #initialize the frontier as stack
    nodeQueue = util.Queue()
    #add initial state of problem th frontier
    nodeQueue.push((problem.getStartState(),[]))

    #loop do #if the frontier is empty then return failure
    while not nodeQueue.isEmpty():
        #chose a node and remove it from the frontier
        currState, actions = nodeQueue.pop()

        #if the node contains a goal state then return the coreesponding solution
        if problem.isGoalState(currState):
            return actions
        
        #add the node state to the explored set
        if currState not in visited:
            visited.add(currState)

            #for each resulting child from node 
            successors = problem.getSuccessors(currState)
            for successor in successors:
                nextState, direction, pathCost = successor
                #if the child state is not already in the frontier or explored set then add child to the frontier
                if nextState not in visited:
                    nodeQueue.push((nextState,actions+[direction]))

    util.raiseNotDefined()

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
    startStateHeuristic = heuristic(problem.getStartState(),problem)
    startStateF = 0 + startStateHeuristic
    nodePriorityQueue.push((problem.getStartState(),[],startStateF),startStateF)
    
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
                nextStateF = nextCost + nextStateHeuristic
                
                # if the child state is not already in the frontier or explored set then
                if nextState not in visited:
                    
                    # add child to the frontier
                    nodePriorityQueue.update((nextState,actions+[action],nextCost),nextStateF)
        
    return[]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


