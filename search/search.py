#!/usr/bin/python
# -*- coding: utf-8 -*-
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
    return [
        s,
        s,
        w,
        s,
        w,
        w,
        s,
        w,
        ]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    # State varibale contains two variables:
        # 1. Coordinates of the current state.
        # 2. Direction taken till the current state.

    #initialize stack as the fringe list as it is DFS which follows LIFO (Last In First Out)
    dataStructure = util.Stack()
    visited=[] #To track visited nodes


    dataStructure.push((problem.getStartState(),[]))

    # Until the fringe list is empty or goal state is reached, pop the state from the fringe list and add its successors to the fringe list
    while not dataStructure.isEmpty():
        coordinates,direction = dataStructure.pop()


        if not coordinates in visited:
            #if current state is the Goal State, Return the directions from the current state
            if problem.isGoalState(coordinates):
                return direction

            for index in problem.getSuccessors(coordinates):
                #if the coordinates of the successors not in the visited array, then push the successor to the fringe list
                if not index[0] in visited:
                    visited=visited+[coordinates]
                    dataStructure.push((index[0],direction+[index[1]]))


    return []


    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""


    # State varibale contains two variables:
        # 1. Coordinates of the current state.
        # 2. Direction taken till the current state.


    #Initialize queue as the fringe list as it is BFS which follows FIFO (First In First Out)
    queue = util.Queue()
    start = problem.getStartState()
    # print(start)


    if not problem.isGoalState(start):
        queue.push((start, []))

    visited = [] # to track the visited nodes
    actions = [] # to store the directions

     # Until the fringe list is empty or goal state is reached, pop the state from the fringe list and add its successors to the fringe list
    while (not queue.isEmpty()):
        curnode, actions = queue.pop()
        if curnode not in visited:

            # if current state is the Goal State, Return the directions from the current state
            if problem.isGoalState(curnode):
                return actions

            visited.append(curnode)

            children = problem.getSuccessors(curnode)

            for coords, direction, cost in children:
                # if the coordinates of the successors not in the visited array, then push the successor to the fringe list
                if coords not in visited:
                    # add the direction to the visited array before pushing it to the fringe list
                    queue.push((coords, actions + [direction]))
    return []
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # This method calculates the path by finding the least cost path from the source to the goal state.
    # State varibale contains two sets of variables:
        # 1. It contains
                    # 1.1 Coordinates of the current state.
                    # 1.2 Directions taken till the current state.
                    # 1.3 Cost till the current state
        # 2. Cost till the current state which is used by the priority queue to get the least cost path

    # Initialise a priority queue to remove the node with the least cost path
    queue = util.PriorityQueue()
    start = problem.getStartState()

    if not problem.isGoalState(start):
        queue.push((start, [],0),0)

    visited = []
    actions = []

    # Until the fringe list is empty or goal state is reached, pop the least cost state from the fringe list and add its successors to the fringe list
    while (not queue.isEmpty()):
        curnode, actions ,originalcost = queue.pop()
        if curnode not in visited:
            #if current state is the Goal State, Return the directions from the current state
            if problem.isGoalState(curnode):
                return actions

            visited.append(curnode)

            children = problem.getSuccessors(curnode)
            #if the coordinates of the successors not in the visited array, then push the successor to the fringe list
            for coords, direction, cost in children:

                if coords not in visited:
                    #Add the cost of the curent node with the cost of the path till the parents node, and push the current state with the final cost to the fringe list
                    queue.push((coords, actions + [direction],cost+originalcost),cost+originalcost)
    return []
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # This method calculates the path by finding the priority f(n) which is the sum of g(n) + h(n) where g(n) is the least cost path to the current node
    # and h(n) is the heuristic function
    # State varibale contains two sets of variables:
        # 1. It contains
                    # 1.1 Coordinates of the current state.
                    # 1.2 Directions taken till the current state.
                    # 1.3 Cost till the current state
        # 2. Cost till the current state added with a huerustic function which is used by the priority queue to get the least cost path

    # Initialise a priority queue to remove the node with the least cost path
    queue = util.PriorityQueue()
    start = problem.getStartState()
    # print(start)
    if not problem.isGoalState(start):
        queue.push((start, [], 0), 0)

    visited = []
    actions = []

    # Until the fringe list is empty or goal state is reached, pop the least cost state from the fringe list and add its successors to the fringe list
    while (not queue.isEmpty()):
        curnode, actions, originalcost = queue.pop()

        if curnode not in visited:

            #if current state is the Goal State, Return the directions from the current state
            if problem.isGoalState(curnode):
                return actions

            visited.append(curnode)

            children = problem.getSuccessors(curnode)

            #if the coordinates of the successors not in the visited array, then push the successor to the fringe list
            for coords, direction, cost in children:

                if coords not in visited:

        #Add the cost of the curent node with the cost of the path till the parents node and the heuristic function, and push the current state with the final cost to the fringe list

                    hs=heuristic(coords,problem)
                    queue.push((coords, actions + [direction], cost + originalcost), cost + originalcost + hs )
    return []
#heuristic(curnode[0],problem)



# Abbreviations

bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
