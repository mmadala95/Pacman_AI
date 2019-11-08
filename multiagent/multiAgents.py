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


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #initialise the score to be returned as zero
        score = 0

        #find the distance of every food particle from the current position of pacman
        foodlist=[manhattanDistance(food,newPos) for food in newFood.asList() ]
        if len(foodlist)==0:
            score = 0
        else:
            score = min(foodlist)

        if(currentGameState.getFood().count()!=newFood.count()):
            score = 0

        #calculate the distance of every ghost from the current position of pacman
        ghostlist=[manhattanDistance(ghost.getPosition(),newPos) for ghost in newGhostStates]
        ghostpos=min(ghostlist)

        #if the ghost is not very close to the pacman, decrease the score with propotion to the distance
        if ghostpos>=3:
            closestghost=-10/ghostpos
        #if the ghost is very near to the pacman, decrease the score drastically so that this direction is avoided
        else:
            closestghost=-1000
        return closestghost-10*score


def scoreEvaluationFunction(currentGameState):
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

    def max_turn(self,gameState,cur_depth):
        actions=gameState.getLegalActions(0)
        if(cur_depth>self.depth or gameState.isWin() or gameState.isLose() or not actions) :
            return ((self.evaluationFunction(gameState),))
        cost=[]
        for action in actions:
            successor=gameState.generateSuccessor(0,action)
            cost.append((self.min_turn(successor,cur_depth,1)[0],action))

        return max(cost)

    def min_turn(self,gameState,cur_depth,index):
        actions=gameState.getLegalActions(index)
        if(gameState.isLose() or gameState.isWin() or not actions):
            return ((self.evaluationFunction(gameState),))
        cost=[]
        for action in actions:
            succesor = gameState.generateSuccessor(index, action)
            if (gameState.getNumAgents()-index-1)==0:
               cost.append((self.max_turn(succesor,cur_depth+1)[0],action))
            else:
                cost.append((self.min_turn(succesor,cur_depth,index+1)[0],action))
        return min(cost)


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"






        value=self.max_turn(gameState,1)
        # print value
        return value[1]
        util.raiseNotDefined()



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_turn(self,gameState,cur_depth,alpha=float('-inf'),beta=float('inf')):

        actions=gameState.getLegalActions(0)
        if(cur_depth>self.depth or gameState.isWin() or gameState.isLose() or not actions) :
            return (self.evaluationFunction(gameState), )

        val=float('-inf')
        return_action=None
        for action in actions:
            successor=gameState.generateSuccessor(0,action)
            newVal=self.min_turn(successor, cur_depth, 1, alpha, beta)[0]
            # v=max([v,(self.min_turn(successor, cur_depth, 1, alpha, beta)[0],action)])
            if(val<newVal):
                val=newVal
                return_action=action

            if val > beta :
                return val,return_action
            if(alpha<val):
                alpha=val

        return val,return_action

    def min_turn(self,gameState,cur_depth,index,alpha,beta):
        actions=gameState.getLegalActions(index)
        if(gameState.isLose() or gameState.isWin() or not actions):
            return (self.evaluationFunction(gameState), )

        val=float('inf')
        return_action=None
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            if (gameState.getNumAgents()-index-1)==0:
                newval=self.max_turn(successor, cur_depth+1, alpha, beta)[0]
                # v = min([beta, (self.max_turn(successor, cur_depth+1, alpha, beta)[0],action)])
                if val>newval:
                    val=newval
                    return_action=action

            else:
                newval2=self.min_turn(successor, cur_depth, index+1 ,alpha, beta)[0]
                # v = min([beta, (self.min_turn(successor, cur_depth, index+1 ,alpha, beta)[0],action)])
                if val>newval2:
                    val=newval2
                    return_action=action
            if val < alpha:
                return val, return_action
            if (beta > val):
                beta = val

        return val,return_action

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value = self.max_turn(gameState, 1)
        # print value
        return value[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def max_turn(self, gameState, cur_depth):

        actions = gameState.getLegalActions(0)
        if (cur_depth > self.depth or gameState.isWin() or gameState.isLose() or not actions):
            return (self.evaluationFunction(gameState), )

        val = float('-inf')
        return_action = None
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            newVal = self.min_turn(successor, cur_depth, 1)[0]
            # v=max([v,(self.min_turn(successor, cur_depth, 1, alpha, beta)[0],action)])
            if (val < newVal):
                val = newVal
                return_action = action

        return val, return_action

    def min_turn(self, gameState, cur_depth, index):
        actions = gameState.getLegalActions(index)
        if (gameState.isLose() or gameState.isWin() or not actions):
            return (self.evaluationFunction(gameState), )

        val = 0
        return_action = None
        for action in actions:
            totalactions=len(actions)
            successor = gameState.generateSuccessor(index, action)
            if (gameState.getNumAgents() - index - 1) == 0:
                newval = self.max_turn(successor, cur_depth + 1)[0]
                val =val + (((1.0)/totalactions)*newval)

            else:
                newval = self.min_turn(successor, cur_depth, index + 1)[0]
                val = val + (((1.0) / totalactions) *newval)
        return val, return_action


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        value = self.max_turn(gameState, 1)
        # print value
        return value[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
