import argparse
import pickle


def main(args):
  with open(args.db, 'rb') as fd:
    w1 = pickle.load(fd)

  for key, value in w1.iteritems():
    print('{0} -> {1}'.format(key, value))

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('db', help='the database file')
  args = ap.parse_args()
  main(args)
