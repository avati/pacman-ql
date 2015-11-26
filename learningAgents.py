from game import Directions
from game import Agent
import random, util
import pickle
import collections
import math



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

The GameState class is defined in pacman.py and you might want to look into that for
other helper methods, though you don't need to.
"""

def roteLearningFeatureExtractor(state, action, features):
  features[(hash(state), action)] += 1

def lineOfSightFeatureExtractor(state, action, features):
  VERBOSE = False
  TIMER_THRES = 3

  pacman = state.getPacmanState()
  ghosts = state.getGhostStates()

  px = int(pacman.configuration.pos[0])
  py = int(pacman.configuration.pos[1])
  if VERBOSE: print('pacman is at [{0},{1}]'.format(px, py))
  for ghost in ghosts:
    gx = int(ghost.configuration.pos[0])
    gy = int(ghost.configuration.pos[1])
    if VERBOSE: print('ghost is at [{0},{1}]'.format(gx, gy))

    # test ghost with common X coordinate
    if px == gx:
      if VERBOSE: print('same X')
      # test ghost is above pacman
      if py <= gy:
        if VERBOSE: print('up')
        # determine if L.O.S. exists
        los = True
        for y in range(py, gy):
          if state.hasWall(px, y):
            los = False
            break
        # if L.O.S., add feature for malicious or scared ghost
        if los:
          if ghost.scaredTimer < TIMER_THRES:
            features[('mn_{0}'.format(gy - py), action)] += 1
          else:
            features[('sn_{0}'.format(gy - py), action)] += 1

      # test ghost is below pacman
      if py >= gy:
        if VERBOSE: print('down')
        # determine if L.O.S exists
        los = True
        for y in range(gy, py):
          if state.hasWall(px, y):
            los = False
            break
        # if L.O.S., add feature for malicious or scared ghost
        if los:
          if ghost.scaredTimer < TIMER_THRES:
            features[('ms_{0}'.format(py - gy), action)] += 1
          else:
            features[('ss_{0}'.format(py - gy), action)] += 1

    # test ghost with common Y coordinate
    if py == gy:
      if VERBOSE: print('same X')
      # test ghost is right of pacman
      if px <= gx:
        if VERBOSE: print('right')
        # determine if L.O.S. exists
        los = True
        for x in range(px, gx):
          if state.hasWall(x, py):
            los = False
            break

        # if L.O.S., add feature for malicious or scared ghost
        if los:
          if ghost.scaredTimer < TIMER_THRES:
            features[('me_{0}'.format(gx - px), action)] += 1
          else:
            features[('se_{0}'.format(gx - px), action)] += 1

      # test ghost is left of pacman
      if px >= gx:
        if VERBOSE: print('left')
        # determine if L.O.S. exists
        los = True
        for x in range(gx, px):
          if state.hasWall(x, py):
            los = False
            break

        # if L.O.S., add feature for malicious or scared ghost
        if los:
          if ghost.scaredTimer < TIMER_THRES:
            features[('mw_{0}'.format(px - gx), action)] += 1
          else:
            features[('sw_{0}'.format(px - gx), action)] += 1



def makeFeatureExtractor(funcs):
  """
  This creates a feature extractor from a list of functions that are
  sub-feature extractors.
  """
  def featureExtractor(state, action):
    features = collections.Counter()
    for func in funcs:
      func(state, action, features)
    return features
  return featureExtractor


class QLearningAgent(Agent):
  def __init__(self, db='database', save='True', expProb='0.0',
               featureExts='rote'):
    # determine if we are going to save the weights at the end of the game
    if save.lower() in ['true', 'yes']:
      self.save = True
    elif save.lower() in ['false', 'no']:
      self.save = False
    else:
      raise Exception('invalid save specifier {0}'.format(save))

    # determine the exploration probability for this run
    self.explorationProb = float(expProb)

    # get the feature extractors
    funcs = []
    for featureExt in featureExts.split(','):
      if featureExt.lower() == 'rote':
        funcs.append(roteLearningFeatureExtractor)
      elif featureExt.lower() == 'los':
        funcs.append(lineOfSightFeatureExtractor)
      else:
        raise Exception('unsupported feature extractor: {0}'.format(featureExt))
    self.featureExtractor = makeFeatureExtractor(funcs)

    # set the database name
    self.db = db + '.db'

    # if weights are available, read them in
    self.weights = collections.Counter()
    self.load_weights()

    # misc
    self.discount = 1
    self.lastState = None
    self.lastAction = None

    print('db={0} save={1} expProb={2}'.format(self.db, save, expProb))

  def registerInitialState(self, state):
    self.lastState = None
    self.lastAction = None
    self.numIters = 0

  # Return the Q function associated with the weights and features
  def getQ(self, state, action):
    score = 0
    for f, v in self.featureExtractor(state, action).iteritems():
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
      return max((self.getQ(state, action), action)
                 for action in self.actions(state))[1]

  def actions(self, state):
    acts = state.getLegalActions()
    #acts.remove(Directions.STOP)
    return acts

  def getStepSize(self):
    return 1.0 / math.sqrt(self.numIters)

  def incorporateFeedback(self, state, action, reward, newState):
    if newState != None:
      v_opt = max(self.getQ(newState, act) for act in self.actions(newState))
    else:
      v_opt = 0

    q_opt = self.getQ(state, action)

    scale = self.getStepSize() * (q_opt - (reward + self.discount * v_opt))

    for f, v in self.featureExtractor(state, action).iteritems():
      self.weights[f] -= scale * v

  def observationFunction(self, state):
    if not self.lastState is None:
      reward = state.getScore() - self.lastState.getScore()
      self.incorporateFeedback(self.lastState, self.lastAction, reward, state)
    return state

  def load_weights(self):
    try:
      self.weights = collections.Counter(pickle.load(open(self.db, 'rb')))
      print('Loaded {0} weights from {1}'.format(len(self.weights), self.db))
    except:
      print('Fresh training')

  def save_weights(self):
    print('Saving {0} weights to {1}'.format(len(self.weights), self.db))
    pickle.dump(self.weights, open(self.db, 'wb'))

  def final(self, state):
    pass  # nothing to do on final state

  def done(self):
    if self.save:
      self.save_weights()
