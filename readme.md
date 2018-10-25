# Pacman MultiAgents AI
Submission for University of Toronto CSC384 - Introduction to Artificial Intelligence Assignment 2 Multiagents.

## Reflex Agent
```
python pacman.py -p ReflexAgent -l testClassic
python pacman.py --frameTime 0 -p ReflexAgent -k 1
python pacman.py --frameTime 0 -p ReflexAgent -k 2
python autograder.py -q q1

```
State-action pair evaluation (i.e. how good it is to perform this action in this state), which added penalties or reward to an action evaluation based on the current state and the result of this action.

## MiniMax Search
```
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python autograder.py -q q2
```
MiniMax implementation for Pacman and multiple ghosts. The search is bounded by a depth, which signifies how many moves Pacman move for each iteration.

## Alpha-Beta Pruning
```
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
python autograder.py -q q3
```
MiniMax with Alpha-Beta pruning, which increases the efficiency significantly. Although result is the same as MiniMax, depth 3 with Alpha-Beta pruning will run as fast as depth 2 of naive MiniMax search.

## ExpectiMax Search
```
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
python autograder.py -q q4
```
ExpectiMax implementation on Pacman problem, which is also bounded by depth.

## Evaluation Function
```
python autograder.py -q q5
```
Instead of valuating actions on a state like the previous evaluation function on ReflexAgent, this evaluation function evaluates a state only.  
This implementation of the function takes into account three main features that is important for a Pacman game:
* Distances to food
* Distances to power pellet / capsules
* Distances from ghosts  
By multiplying these features with their own weight, a linear combination is formed which is the evaluation of a state.