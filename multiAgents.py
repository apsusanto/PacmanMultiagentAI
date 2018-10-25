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

        currentPos = currentGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()
        currentFoodPosition = currentFood.asList()
        currentGhostStates = currentGameState.getGhostStates()
        currentGhostPositions = [ghost.getPosition() for ghost in currentGhostStates]
        currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
        currentCapsules = currentGameState.getCapsules()

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodPosition = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = [ghost.getPosition() for ghost in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsules = successorGameState.getCapsules()

        score = 0


        # Pacman ate a capsule
        if len(newCapsules) < len(currentCapsules):
            score += 15

        if min(util.manhattanDistance(newPos, capsule) for capsule in newCapsules) < min(util.manhattanDistance(newPos, capsule) for capsule in currentCapsules):
            score += 10
        
        # Pacman ate a food
        if len(newFood.asList()) < len(currentGameState.getFood().asList()):
            score += 15
        
        #Add if closer to food
        if min(util.manhattanDistance(newPos, food) for food in newFoodPosition) < min(util.manhattanDistance(currentPos, food) for food in currentFoodPosition):
            score += 20
        
        # Any ghosts are scared, get closer to ghost
        ghostScore = 20 if min(util.manhattanDistance(newPos, ghost) for ghost in newGhostPositions) > min(util.manhattanDistance(currentPos, ghost) for ghost in currentGhostPositions) else -10
        if sum(newScaredTimes) > 0:
            ghostScore *= -1
        
        score += ghostScore

        score -= 5 if action == Directions.STOP else 0

        return successorGameState.getScore() - currentGameState.getScore() + score

            

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
        numAgents = gameState.getNumAgents()

        def minimaxHelper(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or (depth == self.depth and agentIndex == 0):
                return None, self.evaluationFunction(gameState)

            best_val = float("-inf") if agentIndex == 0 else float("inf")

            nextDepth = depth + 1 if agentIndex == (numAgents - 1) else depth

            for move in gameState.getLegalActions(agentIndex):
                nextGameState = gameState.generateSuccessor(agentIndex, move)

                _, next_val = minimaxHelper(nextGameState, nextDepth, (agentIndex + 1) % numAgents)
                
                if (agentIndex == 0 and best_val < next_val) or (agentIndex > 0 and best_val > next_val):
                    best_val, best_move = next_val, move

            return best_move, best_val

        return minimaxHelper(gameState, 0, 0)[0]

            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def alphaBeta(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose() or (depth == self.depth and agentIndex == 0):
                return None, self.evaluationFunction(gameState)

            best_move = None
            best_val = float("-inf") if agentIndex == 0 else float("inf")

            nextDepth = depth + 1 if agentIndex == (numAgents - 1) else depth

            for move in gameState.getLegalActions(agentIndex):
                nextGameState = gameState.generateSuccessor(agentIndex, move)

                _, next_val = alphaBeta(nextGameState, nextDepth, (agentIndex + 1) % numAgents, alpha, beta)
                
                if agentIndex == 0:
                    if best_val < next_val:
                        best_move, best_val = move, next_val
        
                    alpha = max(alpha, best_val)
                    if alpha >= beta:
                        return best_move, best_val
                else:
                    if best_val > next_val:
                        best_move, best_val = move, next_val

                    beta = min(beta, best_val)
                    if beta <= alpha:
                        return best_move, best_val

            return best_move, best_val

        return alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        numAgents = gameState.getNumAgents()

        def expectimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or (depth == self.depth and agentIndex == 0):
                return None, self.evaluationFunction(gameState)

            best_move = None
            best_val = float("-inf") if agentIndex == 0 else 0

            nextDepth = depth + 1 if agentIndex == (numAgents - 1) else depth

            actions = gameState.getLegalActions(agentIndex)

            for move in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, move)

                _, next_val = expectimax(nextGameState, nextDepth, (agentIndex + 1) % numAgents)
                if agentIndex == 0:
                    if best_val < next_val:
                        best_move, best_val = move, next_val
                else:
                    best_val += (1.0 / len(actions)) * next_val
            return best_move, best_val

        return expectimax(gameState, 0, 0)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    ghostPositions = [ghost.getPosition() for ghost in currentGameState.getGhostStates()]
    scaredTimes = [ghost.scaredTimer for ghost in currentGameState.getGhostStates()]
    foodPositions = currentGameState.getFood().asList()
    capsulePositions = currentGameState.getCapsules()
    pos = currentGameState.getPacmanPosition()

    foodDistances = [1.0 / util.manhattanDistance(pos, food) for food in foodPositions]
    capsuleDistances = [1.0 / util.manhattanDistance(pos, capsule) for capsule in capsulePositions]
    ghostDistances = [util.manhattanDistance(pos, ghost) for ghost in ghostPositions]

    avg = lambda x: float(sum(x)) / len(x)
    foodScore = 10 * avg(foodDistances) if foodDistances else float("inf")

    if capsuleDistances:
        capsuleScore = 10 * avg(capsuleDistances) * (1 if sum(scaredTimes) == 0 else -1)
    else:
        capsuleScore = 0
    
    if ghostDistances:
        ghostScore = (20.0 / min(ghostDistances)) if sum(scaredTimes) > 0 else min(ghostDistances) / 2.0
    else:
        ghostScore = 0

    return score + 1.5 * foodScore + capsuleScore + 0.9 * ghostScore

# Abbreviation
better = betterEvaluationFunction

