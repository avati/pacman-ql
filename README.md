# pacman-ql
Pacman AI agent, learns to play with QLearning

## How to run the Q-Learning Pacman Agent
Example:
```bash
python pacman.py -g DirectionalGhost -n 10 -q -p QLearningAgent -a db='weights',save='True',expProb='1.0',featureExt='rote'
```
This runs the QLearningAgent for Pacman.
There are ten games played (`-n 10`) and there is no graphics (`-q`).
The agent arguments denoted by `-a` specify the weights database name.
The feature extractor is specified with the `featureExt` agent argument (`rote` in this example).
The options specify to save the weights after each game and to use a `1.0` probability of exploration.

Note:
Remove the `-q` to watch the game play with graphics. Or exchange the `-q` with a `--frameTime=0.01` to watch it play in hyper speed.

Note:
To play without modifying the weights database, just set `save=False` in the agent arguments.
Also set the `expProb=0.0` so that Pacman acts as best he knows how.