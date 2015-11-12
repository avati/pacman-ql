from util import manhattanDistance
from game import Directions
import random, util
import pickle
import collections
import math

from game import Agent

class QLearningAgentBase(Agent):
  def __init__(self):
    self.discount = 1
    try:
      self.weights = pickle.load(open('weights.db', 'rb'))
      print "Resuming with", str(len(self.weights)), "weights"
    except:
      print "Fresh training"
      self.weights = collections.Counter()
    self.lastState = None
    self.lastAction = None

  def registerInitialState(self, state):
    self.lastState = None
    self.lastAction = None
    self.numIters = 0

  def featureExtractor(self, state, action):
    """
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game

    gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

    Directions.STOP:
        The stop direction, which is always legal

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
    
    gameState.isWin():
        Returns True if it's a winning state
    
    gameState.isLose():
        Returns True if it's a losing state

    self.depth:
        The depth to which search should continue

    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    return [((state, action), 1)]

    # Return the Q function associated with the weights and features
  def getQ(self, state, action):
    score = 0
    for f, v in self.featureExtractor(state, action):
      score += self.weights[f] * v
    return score

  def getAction(self, state):
    self.lastState = state
    self.lastAction = self.doGetAction(state)
    return self.lastAction

  def doGetAction(self, state):
    self.numIters += 1
    if random.random() < self.explorationProb:
      return random.choice(self.actions(state))
    else:
      return max((self.getQ(state, action), action) for action in self.actions(state))[1]

  def actions(self, state):
    acts = state.getLegalActions()
    acts.remove(Directions.STOP)
    return acts

  def getStepSize(self):
    return 1.0 / math.sqrt(self.numIters)

  def incorporateFeedback(self, state, action, reward, newState):
    if newState != None:
      v_opt = max(self.getQ(newState, act) for act in self.actions(newState))
    else:
      v_opt = 0

    phi = dict(self.featureExtractor(state, action))
    q_opt = self.getQ(state, action)

    scale = self.getStepSize()*(q_opt - (reward + self.discount*v_opt))

    for k in phi.keys():
      try:
        self.weights[k] -= scale*phi[k]
      except KeyError:
        self.weights[k] = -scale*phi[k]


  def observationFunction(self, state):
    if not self.lastState is None:
      reward = state.getScore() - self.lastState.getScore()
      self.incorporateFeedback(self.lastState, self.lastAction, reward, state)
    return state

  def save_weights(self, state):
    print "Saving", len(self.weights), "weights"
    pickle.dump(self.weights, open('weights.db', 'wb'))


class QLearningTrainer(QLearningAgentBase):
  def __init__(self):
    QLearningAgentBase.__init__(self)
    self.explorationProb = 1.0

  def final(self, state):
    self.save_weights(state)

class QLearningAgent(QLearningAgentBase):
  def __init__(self):
    QLearningAgentBase.__init__(self)
    self.explorationProb = 0.0

  def final(self, state):
#    self.save_weights(state)
    return
