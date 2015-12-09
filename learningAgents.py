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
  stateCopy = state.deepCopy()
  stateCopy.score = 0
  features[(hash(stateCopy), action)] = 1

def los_extractor_helper(features, state, action, scared, px, py, gx, gy):
  DIST=10
  VERBOSE = False
  if VERBOSE:
    print('pacman is at [{0},{1}]'.format(px, py))
    print('ghost is at [{0},{1}]'.format(gx, gy))

  if scared:
    pfx = "scared_los_"
  else:
    pfx = "los_"

  # test ghost with common X coordinate
  if px == gx:
    if VERBOSE: print('same X')
      # test ghost is above pacman
    if gy in range(py, py + DIST + 1):
      if VERBOSE: print('up')
        # determine if L.O.S. exists
      los = True
      for y in range(py, gy):
        if state.hasWall(px, y):
          los = False
          break
      if (los):
        features[(pfx + 'up', action)] = 1

      # test ghost is below pacman
    if gy in range(py - DIST, py + 1):
      if VERBOSE: print('down')
      # determine if L.O.S exists
      los = True
      for y in range(gy, py):
        if state.hasWall(px, y):
          los = False
          break
      if (los):
        features[(pfx + 'down', action)] = 1

  # test ghost with common Y coordinate
  if py == gy:
    if VERBOSE: print('same X')
    # test ghost is right of pacman
    if gx in range(px, px + DIST + 1):
      if VERBOSE: print('right')
      # determine if L.O.S. exists
      los = True
      for x in range(px, gx):
        if state.hasWall(x, py):
          los = False
          break
      if (los):
        features[(pfx + 'right', action)] = 1

    # test ghost is left of pacman
    if gx in range(px - DIST, px + 1):
      if VERBOSE: print('left')
      # determine if L.O.S. exists
      los = True
      for x in range(gx, px):
        if state.hasWall(x, py):
          los = False
          break
      if (los):
        features[(pfx + 'left', action)] = 1

def lineOfSightFeatureExtractor(state, action, features):
  SCARED_THRESH = 3
  pacman = state.getPacmanState()
  ghosts = state.getGhostStates()

  px = int(pacman.configuration.pos[0])
  py = int(pacman.configuration.pos[1])
  for ghost in ghosts:
    gx = int(ghost.configuration.pos[0])
    gy = int(ghost.configuration.pos[1])

    los_extractor_helper(features, state, action, ghost.scaredTimer > SCARED_THRESH, px, py, gx, gy)

def neighboringGhostFeatureExtractor(state, action, features):
  SCARED_THRESH=3
  dist=2
  pacman = state.getPacmanState()
  ghosts = state.getGhostStates()

  px = int(pacman.configuration.pos[0])
  py = int(pacman.configuration.pos[1])
  for ghost in ghosts:
    gx = int(ghost.configuration.pos[0])
    gy = int(ghost.configuration.pos[1])
    if ghost.scaredTimer > SCARED_THRESH:
      pfx = 'scared_ghost_'
    else:
      pfx = 'ghost_'

    if gy in range(py, py + dist + 1) and abs(gx - px) <= (gy - py) and not state.hasWall(px, py + 1):
      features[(pfx + 'up', action)] = 1
    if gy in range(py - dist, py + 1) and abs(gx - px) <= (py - gy) and not state.hasWall(px, py - 1):
      features[(pfx + 'down', action)] = 1
    if gx in range(px, px + dist + 1) and abs(gy - py) <= (gx - px) and not state.hasWall(px + 1, py):
      features[(pfx + 'right', action)] = 1
    if gx in range(px - dist, px + 1) and abs(gy - py) <= (px - gx) and not state.hasWall(px - 1, py):
      features[(pfx + 'left', action)] = 1

def neighboringFoodFeatureExtractor(state, action, features):
  pacman = state.getPacmanState()
  currentFood = state.getFood()

  px = int(pacman.configuration.pos[0])
  py = int(pacman.configuration.pos[1])

  if currentFood[px][py+1] == True:
    features[('food_up', action)] = 1
  if currentFood[px][py-1] == True:
    features[('food_down', action)] = 1
  if currentFood[px+1][py] == True:
    features[('food_right', action)] = 1
  if currentFood[px-1][py] == True:
    features[('food_left', action)] = 1

def neighboringCapsulesFeatureExtractor(state, action, features):
  pacman = state.getPacmanState()
  currentCapsules = state.getCapsules()

  px = int(pacman.configuration.pos[0])
  py = int(pacman.configuration.pos[1])

  if (px, py+1) in currentCapsules:
    features[('capsule_up', action)] = 1
  if (px, py-1) in currentCapsules:
    features[('capsule_down', action)] = 1
  if (px+1, py) in currentCapsules:
    features[('capsule_right', action)] = 1
  if (px-1, py) in currentCapsules:
    features[('capsule_left', action)] = 1

def makeFeatureExtractor(funcs):
  """
  This creates a feature extractor from a list of functions that are
  sub-feature extractors.
  """
  def featureExtractor(state, action):
    quadraticFeatures = False
    features = collections.Counter()
    for func in funcs:
      func(state, action, features)
    if quadraticFeatures:
      keys = sorted(features)
      for (i, left) in enumerate(keys):
        for right in keys[i+1:]:
          features[(left, right)] = features[left]*features[right]
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
    if featureExts.lower() == 'rote':
      funcs = [roteLearningFeatureExtractor]
    else:
      funcs = [lineOfSightFeatureExtractor,
               neighboringGhostFeatureExtractor,
               neighboringFoodFeatureExtractor,
               neighboringCapsulesFeatureExtractor]

    self.featureExtractor = makeFeatureExtractor(funcs)

    # set the database name
    self.db = db + '.db'

    # if weights are available, read them in
    self.numIters = 0
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
    #self.numIters = 0

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
      qa = list((self.getQ(state, action), action)
                for action in self.actions(state))
      maxval = max(qa)[0]
      maxacts = map(lambda (x, y): y, filter(lambda (x, y): x == maxval, qa))
      if state.lastAction in maxacts:
        return state.lastAction
      if state.lastAction != None:
        opp = {'S': 'N', 'N': 'S', 'E': 'W', 'W': 'E'}[state.lastAction]
        if opp in maxacts and len(maxacts) > 1:
          maxacts.remove(opp)
      return random.choice(maxacts)

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

    q_opt = self.getQ(state, action)

    scale = self.getStepSize() * (q_opt - (reward + self.discount * v_opt))

    for f, v in self.featureExtractor(state, action).iteritems():
      self.weights[f] -= scale * v

    #self.normalize_weights()


  def normalize_weights(self):
    s = sum(self.weights.values())
    self.weights = collections.Counter(dict(map(lambda (k, v): (k, v/s), self.weights.iteritems())))

  def observationFunction(self, state):
    state.lastAction = None
    if not self.lastState is None:
      reward = state.getScore() - self.lastState.getScore()
      self.incorporateFeedback(self.lastState, self.lastAction, reward, state)
      state.lastAction = self.lastAction
    return state

  def load_weights(self):
    try:
      (self.numIters, w) = pickle.load(open(self.db, 'rb'))
      self.weights = collections.Counter(w)
      print('Loaded {0} weights from {1}. numIters={2}'.format(len(self.weights), self.db, self.numIters))
    except:
      print('Fresh training')

  def save_weights(self):
    print('Saving {0} weights to {1}'.format(len(self.weights), self.db))
    pickle.dump((self.numIters, self.weights), open(self.db, 'wb'))

  def final(self, state):
    self.incorporateFeedback(self.lastState, self.lastAction, state.data.scoreChange, None)

  def done(self):
    if self.save:
      #self.normalize_weights()
      self.save_weights()
