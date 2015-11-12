#!/bin/sh

BATCHSIZE=10
QUIET=-q

python pacman.py -p QLearningTrainer -g DirectionalGhost -n $BATCHSIZE $QUIET
