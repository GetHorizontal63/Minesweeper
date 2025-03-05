# Minesweeper AI Solver with Evolutionary Learning
An evolutionary Minesweeper solver that learns optimal strategies through natural selection. Agents evolve by playing thousands of games with increasing difficulty. The system uses constraint solving, pattern matching, and combines tournament selection with adaptive mutation to discover optimal decision parameters.

## Overview
This project implements an AI-powered Minesweeper solver that uses evolutionary algorithms to develop increasingly effective game-playing strategies. The solver consists of:
- An evolutionary training system that progressively increases difficulty as agents improve
- A constraint solver for logical deduction of safe moves and mine locations
- A pattern matcher that recognizes common Minesweeper patterns
- Adaptive mutation and crossover mechanisms for genetic diversity

## Features
- **Evolutionary Training**: The AI learns through generations of simulated gameplay, with the best agents passing their strategies to the next generation.
- **Adaptive Difficulty**: Training starts with small 8x8 grids and gradually increases to 16x16 with higher mine density.
- **Multiple Decision Criteria**: The AI considers various factors like revealed neighbors, edge preference, mine density, and pattern recognition.
- **Safe First Move**: Guarantees a safe first move, just like in traditional Minesweeper.
- **Progress Visualization**: Plots fitness, win rates, deaths, and other metrics over generations.

## How It Works
1. **Population Initialization**: Creates a population of AI agents with random weights for different heuristics.
2. **Fitness Evaluation**: Each agent plays multiple Minesweeper games, with performance tracked.
3. **Selection & Reproduction**: The best agents are selected to produce the next generation through crossover and mutation.
4. **Progressive Difficulty**: Once agents master a difficulty level, the board size or mine density increases.
5. **Stagnation Prevention**: Introduces diversity when improvement plateaus.

## Evolutionary Methodology
The system employs:
- Tournament selection for parent selection
- Three crossover types (uniform, single-point, blend)
- Adaptive mutation rates that decrease as training progresses
- Elitism to preserve the best performers
- Immigration to prevent stagnation

## Game Integration
This AI can be integrated with the included Minesweeper game implementation. The game provides a graphical interface with:
- Multiple difficulty levels (Easy to Insane)
- Various board sizes
- Game state visualization

### Game Screenshots

#### Game Menu
![Menu Screenshot](https://github.com/GetHorizontal63/Minesweeper/blob/main/screenshots/Menu.png)

#### Gameplay on Hard Difficulty
![Hard Difficulty Screenshot](https://github.com/GetHorizontal63/Minesweeper/blob/main/screenshots/Large%20Board%2C%20Hard.png)

## Usage
1. Run the main script to start training:
```
python minesweeper_ai.py
```
2. Configure training parameters:
- Population size
- Games per agent
- Grid size and mine density
3. The system will train agents and output performance statistics, with the best AI weights saved for future use.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Pygame (for visual game interface)

## Future Improvements
- Implement more advanced constraint solving algorithms
- Add reinforcement learning options
- Improve pattern recognition capabilities
- Optimize performance for large board sizes
