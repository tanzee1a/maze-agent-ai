# Team 8 Maze Agent

This project implements a maze-solving agent for a 64x64 maze environment with walls, fire hazards, confusion traps, and teleporters.

The main program trains and tests a hybrid maze-solving agent using exploration, Q-learning, and A* path planning.

## Project Files

```

agent.py           # Hybrid agent logic
environment.py     # Maze environment and step logic
train.py           # Main file for running training and testing
maze_reader.py     # Loads maze images, hazards, start, goal, and wall data
maze_printer.py    # Generates static maze/hazard visualizations
visualizer.py      # Saves episode GIFs and PNG visualizations
TestMazes/         # Folder containing alpha, beta, and gamma maze images
runs/              # Output folder created after running train.py
results/           # Output folder created by maze_printer.py

```

## Requirements

Install the required Python packages:

```

pip install numpy pillow

```

## Maze Files

The project expects the maze files to be stored in this structure:

```
TestMazes/
    maze-alpha/
        MAZE_0.png
        MAZE_1.png
    maze-beta/
        MAZE_0.png
        MAZE_1.png
    maze-gamma/
        MAZE_0.png
        MAZE_1.png
````

`MAZE_0.png` contains the wall layout.

`MAZE_1.png` contains the hazard layout.

## How to Run the Agent

Open `train.py` and choose which maze to run by changing this line near the top:

```python
MAZE_NAME = "gamma"   # change: "alpha" / "beta" / "gamma"
````

Then run:

```
python train.py
```

## Running Alpha

To run the alpha maze, set:

```python
MAZE_NAME = "alpha"
```

Then run:

```
python train.py
```

For alpha, the program runs both training and testing.

It also saves the Q-table to:

```
runs/alpha/q_table.npy
```

## Running Beta or Gamma

To run beta, set:

```python
MAZE_NAME = "beta"
```

To run gamma, set:

```python
MAZE_NAME = "gamma"
```

Then run:

```
python train.py
```

For beta and gamma, the program runs test episodes using a fresh agent state.

## Output Files

After running `train.py`, output files are saved in:

```
runs/<maze-name>/
```

For example:

```
runs/alpha/
runs/beta/
runs/gamma/
```

Episode visualizations are saved in:

```
runs/<maze-name>/viz/train/
runs/<maze-name>/viz/test/
```

The program saves both GIF and PNG files for each episode.

## Metrics Printed

The program prints metrics such as:

```
Success rate
Average path length
Average turns
Death rate
Exploration efficiency
Map completeness
First successful episode
```

These results are printed in the terminal after each run.

## Notes

Run all commands from the main project folder.

If the program cannot find a maze file, check that the `TestMazes` folder is in the correct location and that each maze folder contains both `MAZE_0.png` and `MAZE_1.png`.
