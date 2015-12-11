from game import Agent
from game import Directions
from pacman import GameState
import argparse
import pickle


def main(args):
  with open(args.db, 'rb') as fd:
    (iters, weights, counts) = pickle.load(fd)

  rote = len(weights) == len(counts)

  for key, value in weights.iteritems():
    cnt = counts[key] if rote else ''
    print('[{0}]{1} -> {2}'.format(cnt, key, value))
  print('numIters = {0}'.format(iters))


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('db', help='the database file')
  args = ap.parse_args()
  main(args)
