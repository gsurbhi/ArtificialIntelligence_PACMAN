# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        scoreBasedonGhost=0
        distanceToFood = []
        scoreBasedOnFood=0
            
        ghostValue = 0
        for ghost in newGhostStates:
            distance = manhattanDistance(newPos, newGhostStates[0].getPosition())
            if distance > 0:
                if ghost.scaredTimer > 1:
                    ghostValue += 10 / distance
                else:
                    ghostValue -= 1.0 / distance
        scoreBasedonGhost += ghostValue
        
        distanceToFood = map(lambda x: 1.0 / manhattanDistance(x, newPos), newFood.asList())

 
        if distanceToFood:
            scoreBasedOnFood = max(distanceToFood)
        
        numberOfCapsulesLeft = len(successorGameState.getCapsules())
        scoreBasedonCapsules= -0.5*numberOfCapsulesLeft
        
        return   successorGameState.getScore()*5+scoreBasedonGhost+scoreBasedOnFood+scoreBasedonCapsules

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
        "*** YOUR CODE HERE ***"

        max_value = float('-inf')
        actions = gameState.getLegalActions(0)
        minimax_action = None
        for action in actions:
            value = self.minValue(gameState.generateSuccessor(0,action),self.depth,1)
            if value > max_value:
                minimax_action = action
                max_value = value
        return minimax_action
        #util.raiseNotDefined()

    
    def maxValue(self,gameState,depth):

        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(0)#Pacman action
        v = float('-inf')
        for action in actions:
            successorforPacman = gameState.generateSuccessor(0,action)
            v= max(v,self.minValue(successorforPacman,depth,1))
            
        return v

    def minValue(self,gameState,depth,agent_index):
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)
        v = float('inf')    
        actions = gameState.getLegalActions(agent_index)#Ghost action
        TotalAgents = gameState.getNumAgents() - 1
        successors = [gameState.generateSuccessor(agent_index, action) for action in actions]
        
        for successor in successors:
            if (agent_index == TotalAgents):
                v = min(v,self.maxValue(successor, depth-1))
            else:
                v = min(v,self.minValue(successor, depth,agent_index + 1))
        
        return v

    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        max_value = float('-inf')
        actions = gameState.getLegalActions(0)
        alpha = float('-inf')
        beta  = float('inf')
        return_action = None
        
        for action in actions:
            value = self.minValue(gameState.generateSuccessor(0,action),alpha,beta,0,1)
            if value > alpha:
                return_action = action
                alpha = value
                
        return return_action
    
    
    def maxValue(self,gameState,alpha,beta,depth):


        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        v = float('-inf')     
        actions = gameState.getLegalActions(0)#Pacman action
        for action in actions:
            successorforPacman = gameState.generateSuccessor(0,action)
            v = max(v,self.minValue(successorforPacman,alpha,beta,depth,1))
            if v > beta:
                return v
            alpha = max(alpha,v)
        
        return v

    def minValue(self,gameState,alpha,beta,depth,agent_index):
        
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        v = float('inf')

        TotalAgents = gameState.getNumAgents() - 1
        
        actions = gameState.getLegalActions(agent_index)
       
        for action in actions:
       
            successor = gameState.generateSuccessor(agent_index, action)
            if agent_index == TotalAgents:
                
                v = min(v,self.maxValue(successor,alpha,beta,depth+1))
            else:
                v = min(v,self.minValue(successor,alpha,beta,depth,agent_index + 1))
                  
            if v < alpha:
                return v
            beta = min(beta, v)
        
        return v
        #util.raiseNotDefined()

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
        max_value = float('-inf')
        expectimax_action = None
        for action in gameState.getLegalActions(0): 
            value = self.calculateMin(gameState.generateSuccessor(0,action),1,self.depth)
            if value > max_value:
                expectimax_action = action
                max_value = value
        return expectimax_action
    
    def calculateMax(self,gamestate, current_depth):
        actions = gamestate.getLegalActions(0)#Pacman action

        if current_depth ==0 or gamestate.isWin() or  gamestate.isLose():
            return self.evaluationFunction(gamestate)

        v = []
        for action in actions:
            successor = gamestate.generateSuccessor(0, action)
            v.append(self.calculateMin(successor, 1, current_depth))

        return max(v)

    def calculateMin(self,gamestate,agent_index, current_depth):
        
        actions_for_ghost = gamestate.getLegalActions(agent_index) #Ghost action
        TotalAgents = gamestate.getNumAgents() - 1
        if gamestate.isLose()or gamestate.isWin() or current_depth ==0:
            return self.evaluationFunction(gamestate)

        successors = [gamestate.generateSuccessor(agent_index, action) for action in actions_for_ghost]

        v = []
        for successor in successors:
            if agent_index == TotalAgents:
                v.append(self.calculateMax(successor, current_depth -1))
            else:
                v.append(self.calculateMin(successor, agent_index + 1, current_depth))
        average_prob = float(sum(v)/len(v))
        
        return average_prob

    
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodStates = currentGameState.getFood()
    scoreBasedonGhost=0
    scoreBasedonFood=0
    scoreBasedonCapsules=0    
    ghostValue = 0
    
    for ghost in ghostStates:
        distance = manhattanDistance(position, ghostStates[0].getPosition())
        if distance > 0:
            if ghost.scaredTimer > 1:
                ghostValue += 10 / distance
            else:
                ghostValue -= 1.0 / distance
    scoreBasedonGhost += ghostValue
    
    distancesToFood = [manhattanDistance(position, x) for x in foodStates.asList()]
        
    if len(distancesToFood):
        scoreBasedonFood = 1/min(distancesToFood)
    
    
    numberOfCapsulesLeft = len(currentGameState.getCapsules())
    scoreBasedonCapsules= -0.5*numberOfCapsulesLeft
 
 
    return   score +scoreBasedonGhost+scoreBasedonFood+scoreBasedonCapsules
                  
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

