import sys
import os
import random
import numpy as np
import pandas as pd
import time
import importlib.util
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import math
from collections import defaultdict, deque

# Path to your Minesweeper game
GAME_PATH = r"%USERPROFILE%\PycharmProjects\minesweeper\.venv\Scripts\minesweeper.py"
# Path for exporting data
EXPORT_PATH = r"%USERPROFILE%\PycharmProjects\minesweeper"

# Ensure export directory exists
os.makedirs(EXPORT_PATH, exist_ok=True)

# Import the minesweeper game module
spec = importlib.util.spec_from_file_location("minesweeper", GAME_PATH)
minesweeper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(minesweeper)

# Access the Minesweeper class from the imported module
Minesweeper = minesweeper.Minesweeper


class SafeFirstMoveGameWrapper:
    """Wrapper for Minesweeper that guarantees a safe first move."""

    def __init__(self, game):
        self.game = game
        self.first_move_made = False

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.game, attr)

    def reveal(self, row, col):
        """Override reveal to ensure first move is safe."""
        if not self.first_move_made:
            self.first_move_made = True

            # If first move would hit a mine, relocate the mine
            if self.game.board[row][col] == 'X':
                # Find a safe spot to move the mine to
                safe_row, safe_col = self._find_safe_cell()

                # Move the mine
                self.game.board[row][col] = ' '
                self.game.board[safe_row][safe_col] = 'X'

                # Recalculate numbers around affected cells
                self._recalculate_numbers()

        # Perform the regular reveal operation
        return self.game.reveal(row, col)

    def _find_safe_cell(self):
        """Find a cell without a mine to relocate the mine to."""
        rows, cols = len(self.game.board), len(self.game.board[0])

        while True:
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)

            if self.game.board[r][c] != 'X':
                return r, c

    def _recalculate_numbers(self):
        """Recalculate all number indicators after moving a mine."""
        rows, cols = len(self.game.board), len(self.game.board[0])

        # Clear all number cells (keep only mines or empty spaces)
        for r in range(rows):
            for c in range(cols):
                if self.game.board[r][c] not in ['X', ' ']:
                    self.game.board[r][c] = ' '

        # Recalculate numbers
        for r in range(rows):
            for c in range(cols):
                if self.game.board[r][c] != 'X':
                    # Count adjacent mines
                    mine_count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue

                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if self.game.board[nr][nc] == 'X':
                                    mine_count += 1

                    # Set the cell to the count if > 0, otherwise leave it as space
                    if mine_count > 0:
                        self.game.board[r][c] = str(mine_count)


class MinesweeperAI:
    """AI agent that can play Minesweeper using heuristics and constraint solving."""

    def __init__(self, weights=None):
        # Weights for different heuristics
        self.weights = weights if weights is not None else {
            'revealed_neighbor': 1.0,  # Prefer cells with more revealed neighbors
            'numbered_neighbor': 2.0,  # Prefer cells adjacent to numbered cells
            'safe_probability': 5.0,  # Prefer cells with higher probability of being safe
            'edge_preference': 0.5,  # Slight preference for edges vs middle
            'random_factor': 0.2,  # Small random element for exploration
            'time_penalty': 1.0,  # Weight for time penalty
            'clicks_penalty': 1.5,  # Weight for minimizing clicks
            'first_move_center': 3.0,  # Weight for preferring center for first move
            'pattern_match': 2.5,  # Weight for pattern matching
            'constraint_solving': 6.0,  # Weight for constraint solving
            'progress_reward': 4.0,  # Weight for rewarding progress in the game
            'exploration_vs_exploitation': 1.2,  # Balance between exploring and exploiting
            'corner_avoidance': 1.8,  # Avoid corners unless highly confident
            'analyze_frontiers': 3.5,  # Focus on analyzing frontier cells
            'mine_density_awareness': 2.2,  # Consider local mine density
        }
        self.fitness = 0  # To be set during evaluation
        self.last_move = None  # Track the last move made
        self.stats = {
            'deaths': 0,
            'clicks': 0,
            'time': 0,
            'wins': 0,
            'games_played': 0,
            'cells_revealed': 0,
            'progress_percentage': 0,
        }
        # Memory of past game states for learning
        self.game_history = deque(maxlen=5)  # Keep last 5 game states
        self.constraint_solver = ConstraintSolver()
        self.pattern_matcher = PatternMatcher()

    def mutate(self, mutation_rate=0.1, mutation_scale=0.3, mutation_burst=False):
        """Create a mutated copy of this AI with slightly different weights."""
        new_weights = dict(self.weights)

        # If mutation burst is triggered, be more aggressive with mutations
        actual_mutation_rate = mutation_rate * 3 if mutation_burst else mutation_rate
        actual_mutation_scale = mutation_scale * 2 if mutation_burst else mutation_scale

        for key in new_weights:
            # Randomly decide whether to mutate this weight
            if random.random() < actual_mutation_rate:
                # Apply a random change scaled by mutation_scale
                # Use a normal distribution for more natural mutations
                change = random.normalvariate(0, actual_mutation_scale)
                new_weights[key] = max(0.0, new_weights[key] + change)

        return MinesweeperAI(new_weights)

    def crossover(self, other_ai, crossover_type="uniform"):
        """Create a new AI by crossing over weights with another AI."""
        child_weights = {}

        if crossover_type == "uniform":
            # Uniform crossover: each weight has 50% chance from each parent
            for key in self.weights:
                if random.random() < 0.5:
                    child_weights[key] = self.weights[key]
                else:
                    child_weights[key] = other_ai.weights[key]

        elif crossover_type == "single_point":
            # Single-point crossover: take weights before a random point from one parent, rest from other
            keys = list(self.weights.keys())
            split_point = random.randint(1, len(keys) - 1)

            for i, key in enumerate(keys):
                if i < split_point:
                    child_weights[key] = self.weights[key]
                else:
                    child_weights[key] = other_ai.weights[key]

        elif crossover_type == "blend":
            # Blend crossover: take a weighted average of the two parents' weights
            for key in self.weights:
                blend_ratio = random.random()  # Random blend between the two weights
                child_weights[key] = (blend_ratio * self.weights[key] +
                                      (1 - blend_ratio) * other_ai.weights[key])

        return MinesweeperAI(child_weights)

    def choose_move(self, game_state):
        """Choose the next cell to reveal based on game state and constraints."""
        # Extract current game state
        board = game_state['board']
        revealed = game_state['revealed']
        flagged = game_state['flagged']
        rows, cols = len(board), len(board[0])

        # Save the current state to history for learning
        self.game_history.append(copy.deepcopy(game_state))

        # If this is the first move, target the center
        if game_state['first_move']:
            # Calculate center coordinates
            center_r, center_c = rows // 2, cols // 2

            # Try to find an unrevealed cell close to center
            best_dist = float('inf')
            best_cell = None

            for r in range(rows):
                for c in range(cols):
                    if not revealed[r][c] and not flagged[r][c]:
                        # Manhattan distance to center
                        dist = abs(r - center_r) + abs(c - center_c)
                        if dist < best_dist:
                            best_dist = dist
                            best_cell = (r, c)

            if best_cell:
                return (best_cell[0], best_cell[1], 1)  # 1 = left click

        # Try to use constraint solver to find definite safe cells or mines
        constraints_result = self.constraint_solver.solve(game_state)

        if constraints_result['definite_safe']:
            # We found definite safe cells, pick one of them
            r, c = random.choice(constraints_result['definite_safe'])
            return (r, c, 1)  # Left click on safe cell

        if constraints_result['definite_mines']:
            # We found definite mines, flag one of them
            r, c = random.choice(constraints_result['definite_mines'])
            return (r, c, 3)  # Right click to flag

        # Get all unrevealed and unflagged cells
        unrevealed = []
        for r in range(rows):
            for c in range(cols):
                if not revealed[r][c] and not flagged[r][c]:
                    unrevealed.append((r, c))

        if not unrevealed:
            return None  # No moves available

        # Calculate scores for each possible move
        best_score = -float('inf')
        best_moves = []

        for r, c in unrevealed:
            score = self._evaluate_cell(r, c, game_state, constraints_result)

            if score > best_score:
                best_score = score
                best_moves = [(r, c)]
            elif score == best_score:
                best_moves.append((r, c))

        # Choose randomly among the best moves
        r, c = random.choice(best_moves)

        # Determine if we should flag or reveal
        should_flag = self._should_flag(r, c, game_state, constraints_result)
        action = 3 if should_flag else 1  # 1 = left click, 3 = right click

        self.last_move = (r, c, action)
        return (r, c, action)

    def _evaluate_cell(self, row, col, game_state, constraints_result):
        """Evaluate how promising a cell is to reveal."""
        board = game_state['board']
        revealed = game_state['revealed']
        flagged = game_state['flagged']
        rows, cols = len(board), len(board[0])
        first_move = game_state['first_move']

        # Initialize score
        score = 0.0

        # For first move, prefer center
        if first_move:
            center_r, center_c = rows // 2, cols // 2
            center_dist = abs(row - center_r) + abs(col - center_c)
            max_dist = rows + cols
            center_factor = 1.0 - (center_dist / max_dist)
            score += self.weights['first_move_center'] * center_factor

        # Count revealed neighbors
        revealed_neighbors = 0
        numbered_neighbors = 0
        for r in range(max(0, row - 1), min(rows, row + 2)):
            for c in range(max(0, col - 1), min(cols, col + 2)):
                if (r, c) != (row, col) and revealed[r][c]:
                    revealed_neighbors += 1
                    # Check if it's a numbered cell
                    if board[r][c] not in [' ', 'X']:
                        numbered_neighbors += 1

        # Add weighted scores
        score += self.weights['revealed_neighbor'] * revealed_neighbors / 8.0
        score += self.weights['numbered_neighbor'] * numbered_neighbors / 8.0

        # Calculate a basic probability estimate for safety
        safe_prob = self._estimate_safety(row, col, game_state)
        score += self.weights['safe_probability'] * safe_prob

        # Preference for edges (cells with fewer neighbors)
        num_neighbors = self._count_total_neighbors(row, col, rows, cols)
        edge_factor = 1.0 - (num_neighbors / 8.0)  # Higher for edges
        score += self.weights['edge_preference'] * edge_factor

        # Avoid corners unless highly confident
        is_corner = (row in [0, rows - 1]) and (col in [0, cols - 1])
        if is_corner and safe_prob < 0.9:
            score -= self.weights['corner_avoidance']

        # Pattern matching score
        pattern_score = self.pattern_matcher.evaluate(row, col, game_state)
        score += self.weights['pattern_match'] * pattern_score

        # Constraint solving confidence
        if (row, col) in constraints_result['probable_safe']:
            # Add score based on the confidence level (0.0 to 1.0)
            confidence = constraints_result['probable_safe'][(row, col)]
            score += self.weights['constraint_solving'] * confidence

        # Local mine density awareness
        local_mine_density = self._estimate_local_mine_density(row, col, game_state)
        if local_mine_density < 0.15:  # If local density is low, it's more promising
            score += self.weights['mine_density_awareness'] * (1.0 - local_mine_density * 5)  # Scale to 0-1

        # Check if cell is on a frontier (adjacent to revealed cells)
        is_frontier = revealed_neighbors > 0
        if is_frontier:
            score += self.weights['analyze_frontiers']

        # Balance exploration vs exploitation
        if safe_prob > 0.7:  # Exploit safe cells
            score += self.weights['exploration_vs_exploitation'] * safe_prob
        elif revealed_neighbors == 0 and random.random() < 0.2:  # Occasionally explore unexplored areas
            score += self.weights['exploration_vs_exploitation'] * 0.5

        # Add random factor for exploration
        score += self.weights['random_factor'] * random.random()

        return score

    def _estimate_safety(self, row, col, game_state):
        """Estimate the probability that this cell is safe using multiple heuristics."""
        board = game_state['board']
        revealed = game_state['revealed']
        flagged = game_state['flagged']
        rows, cols = len(board), len(board[0])
        first_move = game_state['first_move']

        # First move is always safe with our wrapper
        if first_move:
            return 1.0

        # Initialize evidence counters
        safe_evidence = 0
        mine_evidence = 0
        total_evidence = 0

        # Check each revealed neighbor for constraints
        for r in range(max(0, row - 1), min(rows, row + 2)):
            for c in range(max(0, col - 1), min(cols, col + 2)):
                if (r, c) != (row, col) and revealed[r][c] and board[r][c] not in [' ', 'X']:
                    # Count flagged and unrevealed neighbors of this numbered cell
                    number = int(board[r][c])
                    flagged_count = 0
                    unrevealed_count = 0
                    unrevealed_cells = []

                    for nr in range(max(0, r - 1), min(rows, r + 2)):
                        for nc in range(max(0, c - 1), min(cols, c + 2)):
                            if (nr, nc) != (r, c):
                                if flagged[nr][nc]:
                                    flagged_count += 1
                                elif not revealed[nr][nc]:
                                    unrevealed_count += 1
                                    unrevealed_cells.append((nr, nc))

                    # If flagged count equals the number, all remaining are safe
                    if flagged_count == number and (row, col) in unrevealed_cells:
                        safe_evidence += 1
                        total_evidence += 1

                    # If (flagged + unrevealed) equals the number, all unrevealed are mines
                    if flagged_count + unrevealed_count == number and unrevealed_count > 0:
                        # Distribute evidence among unrevealed
                        mine_evidence += 1 / unrevealed_count
                        total_evidence += 1

                    # For cells with intermediate cases, calculate probability
                    if flagged_count < number and unrevealed_count > 0:
                        # Probability of being a mine: (number - flagged) / unrevealed
                        local_mine_prob = (number - flagged_count) / unrevealed_count
                        mine_evidence += local_mine_prob
                        total_evidence += 1

        # Add evidence from nearby numbered cells
        for r in range(max(0, row - 2), min(rows, row + 3)):
            for c in range(max(0, col - 2), min(cols, col + 3)):
                if revealed[r][c] and board[r][c] not in [' ', 'X']:
                    # If a revealed cell with a very low number (1) is nearby, it's somewhat safe
                    if board[r][c] == '1':
                        safe_evidence += 0.2
                        total_evidence += 0.2
                    # If a high number is nearby, it's somewhat dangerous
                    elif board[r][c] in ['4', '5', '6', '7', '8']:
                        mine_evidence += 0.3
                        total_evidence += 0.3

        # Compute a safety estimate
        if total_evidence == 0:
            # No evidence, assume base safety level from global mine density
            # Try to guess from the game state
            total_cells = rows * cols
            mine_count = sum(row.count('X') for row in board)
            base_safety = 1.0 - (mine_count / total_cells)
            return base_safety
        elif safe_evidence > 0 and mine_evidence == 0:
            return 1.0  # Cell is definitely safe
        elif mine_evidence > 0 and safe_evidence == 0:
            return 0.0  # Cell is definitely a mine
        else:
            # Calculate weighted probability
            safe_ratio = safe_evidence / total_evidence
            return safe_ratio

    def _should_flag(self, row, col, game_state, constraints_result=None):
        """Decide if we should flag this cell instead of revealing it."""
        # Check if constraint solver found it as a definite mine
        if constraints_result and (row, col) in constraints_result['definite_mines']:
            return True

        # Otherwise use probability estimate
        safe_prob = self._estimate_safety(row, col, game_state)

        # If safety probability is very low, flag it
        return safe_prob < 0.15

    def _count_total_neighbors(self, row, col, rows, cols):
        """Count how many valid neighbors a cell has."""
        count = 0
        for r in range(max(0, row - 1), min(rows, row + 2)):
            for c in range(max(0, col - 1), min(cols, col + 2)):
                if (r, c) != (row, col):
                    count += 1
        return count

    def _estimate_local_mine_density(self, row, col, game_state):
        """Estimate the local mine density around a cell."""
        board = game_state['board']
        revealed = game_state['revealed']
        flagged = game_state['flagged']
        rows, cols = len(board), len(board[0])

        # Count revealed numbers and flagged mines in a 5x5 area
        total_cells = 0
        mine_indicators = 0

        for r in range(max(0, row - 2), min(rows, row + 3)):
            for c in range(max(0, col - 2), min(cols, col + 3)):
                total_cells += 1

                # Count flagged mines
                if flagged[r][c]:
                    mine_indicators += 1

                # Add evidence from numbered cells
                if revealed[r][c] and board[r][c] not in [' ', 'X']:
                    number = int(board[r][c])
                    mine_indicators += number / 8.0  # Scale by max neighbors

        if total_cells == 0:
            return 0.0

        return mine_indicators / total_cells


class ConstraintSolver:
    """Class for solving Minesweeper constraints using logical deduction."""

    def solve(self, game_state):
        """
        Analyze the game state to find definite safe cells and mines.
        Returns a dictionary with definite_safe, definite_mines, and probable_safe.
        """
        board = game_state['board']
        revealed = game_state['revealed']
        flagged = game_state['flagged']
        rows, cols = len(board), len(board[0])

        # Results to return
        result = {
            'definite_safe': [],  # Cells that are definitely safe
            'definite_mines': [],  # Cells that are definitely mines
            'probable_safe': {}  # Cells with a probability of being safe (value is confidence 0-1)
        }

        # Collect all constraints from numbered cells
        constraints = []

        for r in range(rows):
            for c in range(cols):
                if revealed[r][c] and board[r][c] not in [' ', 'X']:
                    number = int(board[r][c])

                    # Get unrevealed neighbors
                    unrevealed_neighbors = []
                    flagged_neighbors = []

                    for nr in range(max(0, r - 1), min(rows, r + 2)):
                        for nc in range(max(0, c - 1), min(cols, c + 2)):
                            if (nr, nc) != (r, c):
                                if flagged[nr][nc]:
                                    flagged_neighbors.append((nr, nc))
                                elif not revealed[nr][nc]:
                                    unrevealed_neighbors.append((nr, nc))

                    # The constraint is: number - flagged = mines in unrevealed
                    remaining_mines = number - len(flagged_neighbors)

                    if unrevealed_neighbors:
                        constraints.append({
                            'cells': set(unrevealed_neighbors),
                            'mines': remaining_mines
                        })

        # Single constraint deductions
        for constraint in constraints:
            cells = constraint['cells']
            mines = constraint['mines']

            # If mines = 0, all cells are safe
            if mines == 0:
                result['definite_safe'].extend(cells)

            # If mines = cells count, all cells are mines
            elif mines == len(cells):
                result['definite_mines'].extend(cells)

            # Otherwise, calculate probability
            else:
                prob_safe = 1.0 - (mines / len(cells))
                for cell in cells:
                    if cell in result['probable_safe']:
                        # Average with existing probability
                        result['probable_safe'][cell] = (result['probable_safe'][cell] + prob_safe) / 2
                    else:
                        result['probable_safe'][cell] = prob_safe

        # Subset-based deductions (more advanced)
        if len(constraints) > 1:
            for i, c1 in enumerate(constraints[:-1]):
                for c2 in constraints[i + 1:]:
                    # If one constraint is a subset of another, we can do deduction
                    if c1['cells'].issubset(c2['cells']):
                        # c1 is subset of c2
                        diff_cells = c2['cells'] - c1['cells']
                        diff_mines = c2['mines'] - c1['mines']

                        if diff_mines == 0:
                            # All cells in the difference are safe
                            result['definite_safe'].extend(diff_cells)
                        elif diff_mines == len(diff_cells):
                            # All cells in the difference are mines
                            result['definite_mines'].extend(diff_cells)

                    elif c2['cells'].issubset(c1['cells']):
                        # c2 is subset of c1
                        diff_cells = c1['cells'] - c2['cells']
                        diff_mines = c1['mines'] - c2['mines']

                        if diff_mines == 0:
                            # All cells in the difference are safe
                            result['definite_safe'].extend(diff_cells)
                        elif diff_mines == len(diff_cells):
                            # All cells in the difference are mines
                            result['definite_mines'].extend(diff_cells)

        # Remove duplicates
        result['definite_safe'] = list(set(result['definite_safe']) - set(result['definite_mines']))
        result['definite_mines'] = list(set(result['definite_mines']))

        return result


class PatternMatcher:
    """Class for recognizing and scoring common Minesweeper patterns."""

    def __init__(self):
        # Define common patterns (relative coordinates and weights)
        # 1-2-1 pattern, 1-2-2-1 pattern, etc.
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize common Minesweeper patterns."""
        patterns = []

        # 1-1 pattern (two 1s adjacent means the non-overlapping cells are safe)
        patterns.append({
            'name': '1-1 pattern',
            'template': [
                {'pos': (0, 0), 'value': '1', 'revealed': True},
                {'pos': (0, 1), 'value': '1', 'revealed': True}
            ],
            'safe_cells': [(-1, 0), (1, 1)],  # Cells that are safe if pattern matches
            'mine_cells': [],  # Cells that are mines if pattern matches
            'weight': 0.8  # Confidence in this pattern
        })

        # 1-2-1 pattern
        patterns.append({
            'name': '1-2-1 pattern',
            'template': [
                {'pos': (0, 0), 'value': '1', 'revealed': True},
                {'pos': (0, 1), 'value': '2', 'revealed': True},
                {'pos': (0, 2), 'value': '1', 'revealed': True}
            ],
            'safe_cells': [(-1, 0), (-1, 2), (1, 0), (1, 2)],
            'mine_cells': [(-1, 1), (1, 1)],
            'weight': 0.9
        })

        return patterns

    def evaluate(self, row, col, game_state):
        """Evaluate if a cell is part of a known pattern and return a score."""
        board = game_state['board']
        revealed = game_state['revealed']
        rows, cols = len(board), len(board[0])

        # Initial score
        score = 0.0

        # Check each pattern
        for pattern in self.patterns:
            # Try to match the pattern with the cell as different positions in the pattern
            for offset_row, offset_col in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                pattern_matches = True

                # Check if all template cells match
                for cell in pattern['template']:
                    r = row + cell['pos'][0] + offset_row
                    c = col + cell['pos'][1] + offset_col

                    # Skip if out of bounds
                    if not (0 <= r < rows and 0 <= c < cols):
                        pattern_matches = False
                        break

                    # Check if cell matches template
                    if cell['revealed'] and (not revealed[r][c] or board[r][c] != cell['value']):
                        pattern_matches = False
                        break

                if pattern_matches:
                    # Check if our target cell is in safe or mine list
                    for safe_pos in pattern['safe_cells']:
                        safe_r = row + safe_pos[0] + offset_row
                        safe_c = col + safe_pos[1] + offset_col

                        if (safe_r, safe_c) == (row, col):
                            score += pattern['weight']

                    for mine_pos in pattern['mine_cells']:
                        mine_r = row + mine_pos[0] + offset_row
                        mine_c = col + mine_pos[1] + offset_col

                        if (mine_r, mine_c) == (row, col):
                            score -= pattern['weight']

        return score


class EvolutionaryTrainer:
    """Trains a population of Minesweeper AIs using evolution with progressive difficulty."""

    # Helper functions that should be defined OUTSIDE the class
    @staticmethod
    def create_training_game(rows=8, cols=8, difficulty="easy", mine_percentage=None):
        """Create a new game instance for training with a safe first move."""
        # If mine_percentage is specified, create a custom game with that percentage
        if mine_percentage is not None:
            class CustomGame(Minesweeper):
                def _get_mine_percentage(self, difficulty):
                    return mine_percentage

            game = CustomGame(rows, cols, difficulty)
        else:
            # Create a standard game
            game = Minesweeper(rows, cols, difficulty)

        # Wrap the game to guarantee safe first move
        return SafeFirstMoveGameWrapper(game)

    def export_training_data(self, filepath=None):
        """Export training history to CSV."""
        if not self.history:
            print("No training data to export")
            return

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(EXPORT_PATH, f"minesweeper_training_{timestamp}.csv")

        # Convert history to DataFrame
        df = pd.DataFrame(self.history)

        # Add weight columns
        weight_keys = list(self.history[0]['weights'].keys())
        for key in weight_keys:
            df[f'weight_{key}'] = [h['weights'].get(key, 0) for h in self.history]

        # Drop the weights dictionary column
        df = df.drop(columns=['weights'])

        # Export to CSV
        df.to_csv(filepath, index=False)
        print(f"Training data exported to {filepath}")

        return filepath

    def plot_training_progress(self, filepath=None, show_plot=False):
        """Plot training progress metrics and save to file without blocking execution."""
        if not self.history:
            print("No training data to plot")
            return

        # Switch to a non-interactive backend to avoid Tkinter issues
        import matplotlib
        matplotlib.use('Agg')  # Use the Agg backend (non-interactive)

        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # Extract data
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        avg_fitness = [h['avg_fitness'] for h in self.history]
        deaths = [h['deaths'] for h in self.history]
        clicks = [h['clicks'] for h in self.history]
        times = [h['time'] for h in self.history]
        wins = [h['wins'] / max(1, h['games_played']) for h in self.history]  # Win rate
        grid_sizes = [f"{h['grid_size'][0]}x{h['grid_size'][1]}" for h in self.history]
        mine_percentages = [h['mine_percentage'] * 100 for h in self.history]  # Convert to percentage

        # Plot fitness
        axs[0, 0].plot(generations, best_fitness, 'b-', label='Best Fitness')
        axs[0, 0].plot(generations, avg_fitness, 'g--', label='Avg Fitness')
        axs[0, 0].set_title('Fitness Over Generations')
        axs[0, 0].set_xlabel('Generation')
        axs[0, 0].set_ylabel('Fitness Score')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot deaths and win rate
        ax1 = axs[0, 1]
        ax1.plot(generations, deaths, 'r-', label='Deaths')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Deaths', color='r')
        ax1.tick_params(axis='y', labelcolor='r')

        ax2 = ax1.twinx()
        ax2.plot(generations, [w * 100 for w in wins], 'g-', label='Win Rate %')
        ax2.set_ylabel('Win Rate %', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax1.set_title('Deaths and Win Rate')

        # Plot clicks
        axs[1, 0].plot(generations, clicks, 'm-')
        axs[1, 0].set_title('Average Clicks per Game')
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Clicks')
        axs[1, 0].grid(True)

        # Plot times
        axs[1, 1].plot(generations, times, 'c-')
        axs[1, 1].set_title('Average Time per Game')
        axs[1, 1].set_xlabel('Generation')
        axs[1, 1].set_ylabel('Time (seconds)')
        axs[1, 1].grid(True)

        # Plot grid size and mine percentage changes
        axs[2, 0].plot(generations, mine_percentages, 'y-')
        axs[2, 0].set_title('Mine Percentage')
        axs[2, 0].set_xlabel('Generation')
        axs[2, 0].set_ylabel('Mine Percentage (%)')
        axs[2, 0].grid(True)

        # Plot grid size changes (this will be more of a step function)
        # Convert grid sizes to a numeric value for plotting
        unique_grid_sizes = list(set(grid_sizes))
        unique_grid_sizes.sort()  # Sort grids by size
        grid_size_values = [unique_grid_sizes.index(size) for size in grid_sizes]

        axs[2, 1].plot(generations, grid_size_values, 'k-')
        axs[2, 1].set_title('Grid Size Progression')
        axs[2, 1].set_xlabel('Generation')
        axs[2, 1].set_yticks(range(len(unique_grid_sizes)))
        axs[2, 1].set_yticklabels(unique_grid_sizes)
        axs[2, 1].grid(True)

        plt.tight_layout()

        # Save the plot
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(EXPORT_PATH, f"minesweeper_training_plot_{timestamp}.png")

        plt.savefig(filepath)
        print(f"Training plot saved to {filepath}")

        # Close the figure to release resources
        plt.close(fig)

        return filepath

    @staticmethod
    def test_best_ai(ai, num_games=20, rows=10, cols=10, mine_percentage=0.15):
        """Test the best AI on a set number of games and report performance metrics."""
        print(f"\nTesting best AI on {num_games} games ({rows}x{cols} grid, {mine_percentage:.1%} mines)...")

        # Stats counters
        wins = 0
        total_moves = 0
        total_time = 0
        total_cells_revealed = 0
        total_cells_total = 0
        dead_games = []

        for game_idx in range(num_games):
            # Create a new game
            game = EvolutionaryTrainer.create_training_game(rows, cols, "easy", mine_percentage)

            # Play the game
            moves_made = 0
            cells_revealed = 0
            start_time = time.time()

            # Count total cells and mines for progress tracking
            total_cells = game.rows * game.cols
            total_mines = sum(row.count('X') for row in game.board)

            game_moves = []  # To record all moves for analysis

            while not game.game_over and moves_made < 500:  # Limit to 500 moves
                # Convert game to state dictionary for AI
                game_state = {
                    'board': game.board,
                    'revealed': game.revealed,
                    'flagged': game.flagged,
                    'first_move': game.first_move,
                    'rows': game.rows,
                    'cols': game.cols
                }

                # Get AI's move
                move = ai.choose_move(game_state)
                if move is None:
                    break

                row, col, action = move
                game_moves.append((row, col, action))

                # Count revealed cells before move
                revealed_before = sum(row.count(True) for row in game.revealed)

                # Apply the move
                if action == 1:  # Left click (reveal)
                    if game.revealed[row][col]:
                        game.chord(row, col)
                    else:
                        game.reveal(row, col)
                elif action == 3:  # Right click (flag)
                    game.toggle_flag(row, col)

                # Count newly revealed cells
                revealed_after = sum(row.count(True) for row in game.revealed)
                cells_revealed += (revealed_after - revealed_before)

                moves_made += 1

            # Calculate game time
            game_time = time.time() - start_time

            # Update stats
            if game.win:
                wins += 1
            else:
                # Record information about failed games for analysis
                dead_games.append({
                    'game_idx': game_idx,
                    'moves_made': moves_made,
                    'cells_revealed': cells_revealed,
                    'total_cells': total_cells,
                    'total_mines': total_mines,
                    'last_move': game_moves[-1] if game_moves else None
                })

            total_moves += moves_made
            total_time += game_time
            total_cells_revealed += cells_revealed
            total_cells_total += (total_cells - total_mines)

            # Print progress
            result = "WIN" if game.win else "LOSS"
            progress = cells_revealed / (total_cells - total_mines) * 100
            print(f"  Game {game_idx + 1}/{num_games}: {result}, "
                  f"Moves: {moves_made}, Time: {game_time:.2f}s, "
                  f"Progress: {progress:.1f}%, Cells: {cells_revealed}/{total_cells - total_mines}")

        # Calculate overall statistics
        win_rate = (wins / num_games) * 100
        avg_moves = total_moves / num_games
        avg_time = total_time / num_games
        avg_cells = total_cells_revealed / num_games
        progress_percentage = total_cells_revealed / total_cells_total * 100

        print("\nTest Results:")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Average Moves per Game: {avg_moves:.2f}")
        print(f"  Average Time per Game: {avg_time:.2f} seconds")
        print(f"  Average Cells Revealed per Game: {avg_cells:.2f}")
        print(f"  Overall Progress: {progress_percentage:.2f}%")

        # Analyze failed games
        if dead_games:
            print("\nFailed Games Analysis:")
            avg_progress_failed = sum(
                g['cells_revealed'] / (g['total_cells'] - g['total_mines']) for g in dead_games) / len(dead_games) * 100
            print(f"  Average Progress in Failed Games: {avg_progress_failed:.2f}%")

            # Check at what point in the game AI tends to fail
            early_failures = len(
                [g for g in dead_games if g['cells_revealed'] / (g['total_cells'] - g['total_mines']) < 0.3])
            mid_failures = len(
                [g for g in dead_games if 0.3 <= g['cells_revealed'] / (g['total_cells'] - g['total_mines']) < 0.7])
            late_failures = len(
                [g for g in dead_games if g['cells_revealed'] / (g['total_cells'] - g['total_mines']) >= 0.7])

            print(f"  Failure Distribution:")
            print(
                f"    Early-game failures (< 30% progress): {early_failures} ({early_failures / len(dead_games) * 100:.1f}%)")
            print(
                f"    Mid-game failures (30-70% progress): {mid_failures} ({mid_failures / len(dead_games) * 100:.1f}%)")
            print(
                f"    Late-game failures (> 70% progress): {late_failures} ({late_failures / len(dead_games) * 100:.1f}%)")

        return {
            'win_rate': win_rate,
            'avg_moves': avg_moves,
            'avg_time': avg_time,
            'avg_cells': avg_cells,
            'progress_percentage': progress_percentage,
            'failed_games': len(dead_games),
            'early_failures': early_failures if dead_games else 0,
            'mid_failures': mid_failures if dead_games else 0,
            'late_failures': late_failures if dead_games else 0
        }

    def __init__(self, population_size=30, elitism=5):
        self.population_size = population_size
        self.elitism = elitism  # Number of top performers to keep unchanged
        self.population = [MinesweeperAI() for _ in range(population_size)]
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_ai = None
        self.history = []  # Store training history

        # Progressive difficulty settings
        self.current_grid_size = (8, 8)  # Starting with 8x8
        self.current_mine_percentage = 0.10  # Starting with 10%
        self.consecutive_success_generations = 0
        self.success_threshold = 15  # Number of generations before increasing difficulty
        self.success_criteria_win_rate = 0.90  # 90% win rate to consider successful
        self.success_criteria_max_deaths = 2  # No more than 2 deaths per 100 games

        # Diversity mechanisms
        self.last_best_fitness = -float('inf')
        self.generations_without_improvement = 0
        self.stagnation_threshold = 10  # After this many generations with no improvement, inject diversity
        self.immigration_rate = 0.1  # Percentage of population to replace with random AIs

    def evolve(self, mutation_rate=0.1, mutation_scale=0.3):
        """Evolve the population to the next generation with adaptive parameters."""
        # Sort by fitness (descending)
        self.population.sort(key=lambda ai: ai.fitness, reverse=True)

        # Save the best performer from this generation
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_ai = MinesweeperAI(dict(self.population[0].weights))  # Create a copy
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        # Check for stagnation and handle it
        stagnation = self.generations_without_improvement >= self.stagnation_threshold
        if stagnation:
            print("Detected stagnation - injecting diversity")

        # Track success for progressive difficulty
        best_win_rate = self.population[0].stats['wins'] / max(1, self.population[0].stats['games_played'])
        best_deaths = self.population[0].stats['deaths']

        if (best_win_rate >= self.success_criteria_win_rate and
                best_deaths <= self.success_criteria_max_deaths):
            self.consecutive_success_generations += 1
            print(f"Success generation {self.consecutive_success_generations}/{self.success_threshold}")
        else:
            self.consecutive_success_generations = 0

        # Record history
        generation_stats = {
            'generation': self.generation,
            'best_fitness': self.population[0].fitness,
            'avg_fitness': np.mean([ai.fitness for ai in self.population]),
            'deaths': self.population[0].stats['deaths'],
            'clicks': self.population[0].stats['clicks'],
            'time': self.population[0].stats['time'],
            'wins': self.population[0].stats['wins'],
            'games_played': self.population[0].stats['games_played'],
            'win_rate': best_win_rate * 100,
            'grid_size': self.current_grid_size,
            'mine_percentage': self.current_mine_percentage,
            'weights': dict(self.population[0].weights)
        }
        self.history.append(generation_stats)

        # Check if we should increase difficulty
        if self.consecutive_success_generations >= self.success_threshold:
            self._increase_difficulty()
            self.consecutive_success_generations = 0

        # Create the next generation
        next_generation = []

        # Always include the best AI ever seen in the next generation
        if self.best_ai is not None:
            next_generation.append(MinesweeperAI(dict(self.best_ai.weights)))  # Create a copy

        # Keep the best performers from current generation (elitism)
        elite = self.population[:self.elitism]
        next_generation.extend([ai.mutate(mutation_rate / 2, mutation_scale / 2) for ai in elite])

        # Calculate how many AIs to add through regular evolution
        immigrants_count = 0
        if stagnation:
            immigrants_count = int(self.immigration_rate * self.population_size)

        regular_count = self.population_size - len(next_generation) - immigrants_count

        # Fill the next part with crossover and mutation
        crossover_types = ["uniform", "single_point", "blend"]
        for _ in range(regular_count):
            # Adaptive crossover: use different types with different weights
            crossover_type = random.choice(crossover_types)

            # Tournament selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()

            # Crossover
            child = parent1.crossover(parent2, crossover_type=crossover_type)

            # Possibly do a mutation burst if stagnating
            mutation_burst = stagnation and random.random() < 0.3

            # Mutation
            child = child.mutate(mutation_rate, mutation_scale, mutation_burst=mutation_burst)

            next_generation.append(child)

        # Add random immigrants if needed
        for _ in range(immigrants_count):
            next_generation.append(MinesweeperAI())  # Add completely random AI

        # If we have too many due to adding the best_ai, trim the excess
        if len(next_generation) > self.population_size:
            next_generation = next_generation[:self.population_size]

        self.population = next_generation
        self.generation += 1

        return self.best_ai

    def _select_parent(self, tournament_size=3):
        """Select a parent using tournament selection."""
        # Select random individuals for the tournament
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))

        # Return the best one
        return max(tournament, key=lambda ai: ai.fitness)

    def _increase_difficulty(self):
        """Increase the difficulty by expanding the grid or increasing mine percentage."""
        print("\n=== INCREASING DIFFICULTY ===")
        print(f"Current settings: {self.current_grid_size} grid with {self.current_mine_percentage:.1%} mines")

        # First, increase grid size
        rows, cols = self.current_grid_size
        if rows < 16:  # Cap at 16x16
            self.current_grid_size = (rows * 2, cols * 2)  # Double both dimensions
            print(f"Grid size increased to: {self.current_grid_size}")
        # If grid size is already at max, increase mine percentage
        elif self.current_mine_percentage < 0.25:  # Cap at 25%
            self.current_mine_percentage += 0.01  # Increase by 1 percentage point
            print(f"Mine percentage increased to: {self.current_mine_percentage:.1%}")

        print("=== DIFFICULTY INCREASED ===\n")

    def evaluate_population(self, game_creator, games_per_ai=5, max_moves=500, headless=True):
        """Evaluate all AIs in the population."""
        total_ais = len(self.population)

        print(f"Evaluating generation {self.generation} ({total_ais} AIs, {games_per_ai} games each)")
        print(f"Current settings: {self.current_grid_size} grid with {self.current_mine_percentage:.1%} mines")

        # Reset statistics for all AIs
        for ai in self.population:
            ai.stats = {
                'deaths': 0,
                'clicks': 0,
                'time': 0,
                'wins': 0,
                'games_played': 0,
                'cells_revealed': 0,
                'progress_percentage': 0,
            }

        # For each AI in the population
        for ai_idx, ai in enumerate(self.population):
            total_score = 0
            total_cells_revealed = 0
            total_total_cells = 0

            # Print progress update
            if ai_idx % 5 == 0:
                print(f"  Evaluating AI {ai_idx}/{total_ais}...")

            # Play multiple games
            for game_idx in range(games_per_ai):
                game = game_creator()  # Create a fresh game
                score, stats = self._play_game(ai, game, max_moves, headless)

                # Update AI stats
                ai.stats['deaths'] += 1 if not stats['win'] else 0
                ai.stats['clicks'] += stats['moves']
                ai.stats['time'] += stats['time']
                ai.stats['wins'] += 1 if stats['win'] else 0
                ai.stats['games_played'] += 1
                ai.stats['cells_revealed'] += stats['cells_revealed']

                # Track total cells for progress calculation
                total_cells_revealed += stats['cells_revealed']
                total_total_cells += stats['total_cells'] - stats['total_mines']

                total_score += score

            # Average the stats
            if games_per_ai > 0:
                ai.stats['clicks'] /= games_per_ai
                ai.stats['time'] /= games_per_ai

                # Calculate progress percentage (how many non-mine cells were revealed on average)
                if total_total_cells > 0:
                    ai.stats['progress_percentage'] = (total_cells_revealed / total_total_cells) * 100

            # Set fitness score
            ai.fitness = total_score / games_per_ai

    def _play_game(self, ai, game, max_moves=500, headless=True):
        """Have the AI play a complete game and return a score and stats."""
        moves_made = 0
        cells_revealed = 0
        start_time = time.time()

        # Count total cells and mines for progress tracking
        total_cells = game.rows * game.cols
        total_mines = sum(row.count('X') for row in game.board)

        while not game.game_over and moves_made < max_moves:
            # Convert game to state dictionary for AI
            game_state = {
                'board': game.board,
                'revealed': game.revealed,
                'flagged': game.flagged,
                'first_move': game.first_move,
                'rows': game.rows,
                'cols': game.cols
            }

            # Get AI's move
            move = ai.choose_move(game_state)
            if move is None:
                break

            row, col, action = move

            # Count revealed cells before move
            revealed_before = sum(row.count(True) for row in game.revealed)

            # Apply the move
            if action == 1:  # Left click (reveal)
                if game.revealed[row][col]:
                    game.chord(row, col)
                else:
                    game.reveal(row, col)
            elif action == 3:  # Right click (flag)
                game.toggle_flag(row, col)

            # Count newly revealed cells
            revealed_after = sum(row.count(True) for row in game.revealed)
            cells_revealed += (revealed_after - revealed_before)

            moves_made += 1

            # If it hit a mine, calculate score with consideration for progress
            if game.game_over and not game.win:
                # Calculate stats
                end_time = time.time()
                game_time = end_time - start_time

                # Base penalty for hitting a mine
                mine_penalty = -50

                # Add some partial credit for progress
                progress_factor = cells_revealed / (total_cells - total_mines)
                progress_score = 300 * progress_factor

                # Calculate final score with penalties
                final_score = mine_penalty + progress_score - moves_made * ai.weights['clicks_penalty']

                return final_score, {
                    'win': False,
                    'moves': moves_made,
                    'cells_revealed': cells_revealed,
                    'time': game_time,
                    'total_cells': total_cells,
                    'total_mines': total_mines
                }

        # Calculate final time
        end_time = time.time()
        game_time = end_time - start_time

        # Calculate score with more nuanced rewards
        score = 0

        # Reward for cells revealed (progressive)
        cells_reward = cells_revealed * 10

        # Calculate progress percentage
        progress_percentage = cells_revealed / (total_cells - total_mines)

        # Extra reward for high progress percentage
        progress_bonus = 0
        if progress_percentage > 0.5:
            progress_bonus = 100 * progress_percentage

        # Bonus for winning
        win_bonus = 0
        if game.win:
            win_bonus = 500

            # Extra bonus for winning efficiently
            efficiency_factor = 1.0 - (moves_made / (2 * (total_cells - total_mines)))
            if efficiency_factor > 0:
                win_bonus += 200 * efficiency_factor

        # Penalties
        move_penalty = moves_made * ai.weights['clicks_penalty']
        time_penalty = game_time * ai.weights['time_penalty']

        # Calculate final score
        score = cells_reward + progress_bonus + win_bonus - move_penalty - time_penalty

        return score, {
            'win': game.win,
            'moves': moves_made,
            'cells_revealed': cells_revealed,
            'time': game_time,
            'total_cells': total_cells,
            'total_mines': total_mines
        }

    def export_training_data(self, filepath=None):
        """Export training history to CSV."""
        if not self.history:
            print("No training data to export")
            return

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(EXPORT_PATH, f"minesweeper_training_{timestamp}.csv")

        # Convert history to DataFrame
        df = pd.DataFrame(self.history)

        # Add weight columns
        weight_keys = list(self.history[0]['weights'].keys())
        for key in weight_keys:
            df[f'weight_{key}'] = [h['weights'].get(key, 0) for h in self.history]

        # Drop the weights dictionary column
        df = df.drop(columns=['weights'])

        # Export to CSV
        df.to_csv(filepath, index=False)
        print(f"Training data exported to {filepath}")

        return filepath


# Define these functions at the module level (outside any class)
def create_training_game(rows=8, cols=8, difficulty="easy", mine_percentage=None):
    """Create a new game instance for training with a safe first move."""
    # If mine_percentage is specified, create a custom game with that percentage
    if mine_percentage is not None:
        class CustomGame(Minesweeper):
            def _get_mine_percentage(self, difficulty):
                return mine_percentage

        game = CustomGame(rows, cols, difficulty)
    else:
        # Create a standard game
        game = Minesweeper(rows, cols, difficulty)

    # Wrap the game to guarantee safe first move
    return SafeFirstMoveGameWrapper(game)


def train_minesweeper_ai(generations=100, population_size=30, games_per_ai=10,
                         initial_grid_size=(8, 8), initial_mine_percentage=0.10,
                         export=True):
    """Train a Minesweeper AI over multiple generations with progressive difficulty."""
    # Create trainer
    trainer = EvolutionaryTrainer(population_size=population_size)
    trainer.current_grid_size = initial_grid_size
    trainer.current_mine_percentage = initial_mine_percentage

    print(f"Starting training for {generations} generations")
    print(f"Population: {population_size}, Games per AI: {games_per_ai}")
    print(f"Initial Grid: {initial_grid_size}, Initial Mine Percentage: {initial_mine_percentage:.1%}")

    start_time = time.time()

    # For adaptive mutation rates
    base_mutation_rate = 0.1
    base_mutation_scale = 0.3

    # Train for specified generations
    for gen in range(generations):
        # Adaptive mutation rates - decrease as training progresses
        current_mutation_rate = base_mutation_rate * (1.0 - min(0.6, gen / generations))
        current_mutation_scale = base_mutation_scale * (1.0 - min(0.4, gen / generations))

        # Create a function to generate training games with our parameters
        rows, cols = trainer.current_grid_size
        mine_percentage = trainer.current_mine_percentage
        game_creator = lambda: create_training_game(rows, cols, "easy", mine_percentage)

        # Evaluate and evolve
        trainer.evaluate_population(game_creator, games_per_ai=games_per_ai)
        best_ai = trainer.evolve(mutation_rate=current_mutation_rate,
                                 mutation_scale=current_mutation_scale)

        # Display progress
        win_rate = best_ai.stats['wins'] / max(1, best_ai.stats['games_played']) * 100
        print(f"Generation {gen + 1}/{generations} complete")
        print(f"  Best fitness: {trainer.best_fitness:.2f}")
        print(f"  Avg clicks: {best_ai.stats['clicks']:.2f}")
        print(f"  Avg time: {best_ai.stats['time']:.2f} seconds")
        print(f"  Deaths: {best_ai.stats['deaths']}/{best_ai.stats['games_played']}")
        print(f"  Win rate: {win_rate:.2f}%")
        print(f"  Progress: {best_ai.stats['progress_percentage']:.2f}%")

        # Save intermediate plot every 10 generations
        if (gen + 1) % 10 == 0 or gen == 0:
            plot_path = os.path.join(EXPORT_PATH, f"training_progress.png")
            trainer.plot_training_progress(filepath=plot_path)

        # Optional: break early if we've reached a good performance on the largest grid
        rows, cols = trainer.current_grid_size
        if (rows >= 16 and cols >= 16 and
                trainer.current_mine_percentage >= 0.20 and
                win_rate > 90 and
                best_ai.stats['deaths'] < 5):
            print("Early stopping: mastered the hardest difficulty")
            break

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Export results
    if export:
        csv_path = trainer.export_training_data()

        # Save final plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_plot_path = os.path.join(EXPORT_PATH, f"final_training_plot_{timestamp}.png")
        trainer.plot_training_progress(filepath=final_plot_path)

        # Save the best AI weights
        best_weights = dict(best_ai.weights)
        weights_path = os.path.join(EXPORT_PATH, f"best_ai_weights_{timestamp}.txt")

        with open(weights_path, 'w') as f:
            f.write("# Best AI Weights\n")
            f.write(f"# Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"# Final grid size: {trainer.current_grid_size}, Mine percentage: {trainer.current_mine_percentage:.1%}\n")
            f.write(f"# Generations: {generations}, Final generation: {trainer.generation}\n")
            f.write(f"# Best fitness: {trainer.best_fitness:.2f}\n\n")

            for key, value in best_weights.items():
                f.write(f"{key} = {value:.4f}\n")

        print(f"Best AI weights saved to {weights_path}")

    return best_ai, trainer


def test_best_ai(ai, num_games=20, rows=10, cols=10, mine_percentage=0.15):
    """Test the best AI on a set number of games and report performance metrics."""
    print(f"\nTesting best AI on {num_games} games ({rows}x{cols} grid, {mine_percentage:.1%} mines)...")

    # Stats counters
    wins = 0
    total_moves = 0
    total_time = 0
    total_cells_revealed = 0
    total_cells_total = 0
    dead_games = []

    for game_idx in range(num_games):
        # Create a new game
        game = create_training_game(rows, cols, "easy", mine_percentage)

        # Play the game
        moves_made = 0
        cells_revealed = 0
        start_time = time.time()

        # Count total cells and mines for progress tracking
        total_cells = game.rows * game.cols
        total_mines = sum(row.count('X') for row in game.board)

        game_moves = []  # To record all moves for analysis

        while not game.game_over and moves_made < 500:  # Limit to 500 moves
            # Convert game to state dictionary for AI
            game_state = {
                'board': game.board,
                'revealed': game.revealed,
                'flagged': game.flagged,
                'first_move': game.first_move,
                'rows': game.rows,
                'cols': game.cols
            }

            # Get AI's move
            move = ai.choose_move(game_state)
            if move is None:
                break

            row, col, action = move
            game_moves.append((row, col, action))

            # Count revealed cells before move
            revealed_before = sum(row.count(True) for row in game.revealed)

            # Apply the move
            if action == 1:  # Left click (reveal)
                if game.revealed[row][col]:
                    game.chord(row, col)
                else:
                    game.reveal(row, col)
            elif action == 3:  # Right click (flag)
                game.toggle_flag(row, col)

            # Count newly revealed cells
            revealed_after = sum(row.count(True) for row in game.revealed)
            cells_revealed += (revealed_after - revealed_before)

            moves_made += 1

        # Calculate game time
        game_time = time.time() - start_time

        # Update stats
        if game.win:
            wins += 1
        else:
            # Record information about failed games for analysis
            dead_games.append({
                'game_idx': game_idx,
                'moves_made': moves_made,
                'cells_revealed': cells_revealed,
                'total_cells': total_cells,
                'total_mines': total_mines,
                'last_move': game_moves[-1] if game_moves else None
            })

        total_moves += moves_made
        total_time += game_time
        total_cells_revealed += cells_revealed
        total_cells_total += (total_cells - total_mines)

        # Print progress
        result = "WIN" if game.win else "LOSS"
        progress = cells_revealed / (total_cells - total_mines) * 100
        print(f"  Game {game_idx + 1}/{num_games}: {result}, "
              f"Moves: {moves_made}, Time: {game_time:.2f}s, "
              f"Progress: {progress:.1f}%, Cells: {cells_revealed}/{total_cells - total_mines}")

    # Calculate overall statistics
    win_rate = (wins / num_games) * 100
    avg_moves = total_moves / num_games
    avg_time = total_time / num_games
    avg_cells = total_cells_revealed / num_games
    progress_percentage = total_cells_revealed / total_cells_total * 100

    print("\nTest Results:")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Average Moves per Game: {avg_moves:.2f}")
    print(f"  Average Time per Game: {avg_time:.2f} seconds")
    print(f"  Average Cells Revealed per Game: {avg_cells:.2f}")
    print(f"  Overall Progress: {progress_percentage:.2f}%")

    # Analyze failed games
    if dead_games:
        print("\nFailed Games Analysis:")
        avg_progress_failed = sum(
            g['cells_revealed'] / (g['total_cells'] - g['total_mines']) for g in dead_games) / len(dead_games) * 100
        print(f"  Average Progress in Failed Games: {avg_progress_failed:.2f}%")

        # Check at what point in the game AI tends to fail
        early_failures = len(
            [g for g in dead_games if g['cells_revealed'] / (g['total_cells'] - g['total_mines']) < 0.3])
        mid_failures = len(
            [g for g in dead_games if 0.3 <= g['cells_revealed'] / (g['total_cells'] - g['total_mines']) < 0.7])
        late_failures = len(
            [g for g in dead_games if g['cells_revealed'] / (g['total_cells'] - g['total_mines']) >= 0.7])

        print(f"  Failure Distribution:")
        print(
            f"    Early-game failures (< 30% progress): {early_failures} ({early_failures / len(dead_games) * 100:.1f}%)")
        print(
            f"    Mid-game failures (30-70% progress): {mid_failures} ({mid_failures / len(dead_games) * 100:.1f}%)")
        print(
            f"    Late-game failures (> 70% progress): {late_failures} ({late_failures / len(dead_games) * 100:.1f}%)")

    return {
        'win_rate': win_rate,
        'avg_moves': avg_moves,
        'avg_time': avg_time,
        'avg_cells': avg_cells,
        'progress_percentage': progress_percentage,
        'failed_games': len(dead_games),
        'early_failures': early_failures if dead_games else 0,
        'mid_failures': mid_failures if dead_games else 0,
        'late_failures': late_failures if dead_games else 0
    }


if __name__ == "__main__":
    # Force matplotlib to use non-interactive backend from the start
    import matplotlib

    matplotlib.use('Agg')

    # Get user input for training parameters
    print("=== Minesweeper AI Training with Progressive Difficulty ===")

    try:
        # Core training parameters
        population_size = int(input("Enter population size (20-50 recommended): ") or "30")
        games_per_ai = int(input("Enter games per AI evaluation (10-100 recommended): ") or "20")
        generations = int(input("Enter max generations (100-1000 recommended): ") or "200")

        # Optional: Set initial difficulty
        initial_rows = int(input("Enter initial grid rows (default 8): ") or "8")
        initial_cols = int(input("Enter initial grid columns (default 8): ") or "8")
        initial_mine_pct = float(input("Enter initial mine percentage (0.05-0.15 recommended): ") or "0.10")

        # Validate inputs
        if population_size < 5:
            print("Population size too small. Using 10.")
            population_size = 10

        if games_per_ai < 5:
            print("Games per AI too few. Using 10.")
            games_per_ai = 10
    except ValueError:
        print("Invalid input. Using default values.")
        population_size = 30
        games_per_ai = 20
        generations = 200
        initial_rows = 8
        initial_cols = 8
        initial_mine_pct = 0.10

    print("\nStarting training with the following parameters:")
    print(f"Population size: {population_size}")
    print(f"Games per AI: {games_per_ai}")
    print(f"Max generations: {generations}")
    print(f"Initial grid: {initial_rows}x{initial_cols}")
    print(f"Initial mine percentage: {initial_mine_pct:.1%}")
    print()

    # Run the training
    best_ai, trainer = train_minesweeper_ai(
        generations=generations,
        population_size=population_size,
        games_per_ai=games_per_ai,
        initial_grid_size=(initial_rows, initial_cols),
        initial_mine_percentage=initial_mine_pct,
        export=True
    )

    # Test the best AI at different difficulty levels
    print("\n=== Testing Best AI at Different Difficulty Levels ===")

    # Test on small grid with low mine density (Easy)
    test_stats_easy = test_best_ai(best_ai, num_games=50, rows=8, cols=8, mine_percentage=0.10)

    # Test on medium grid with medium mine density (Medium)
    test_stats_medium = test_best_ai(best_ai, num_games=30, rows=16, cols=16, mine_percentage=0.15)

    # Test on large grid with high mine density (Hard)
    test_stats_hard = test_best_ai(best_ai, num_games=20, rows=16, cols=16, mine_percentage=0.20)

    # Save comprehensive test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_results_path = os.path.join(EXPORT_PATH, f"ai_comprehensive_test_{timestamp}.txt")

    with open(test_results_path, 'w') as f:
        f.write("# Minesweeper AI Comprehensive Test Results\n")
        f.write(f"# Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Training specifications: {generations} generations, {population_size} population size\n")
        f.write(
            f"# Final training settings: {trainer.current_grid_size} grid, {trainer.current_mine_percentage:.1%} mines\n\n")

        f.write("## Easy Difficulty (8x8, 10% mines)\n")
        for key, value in test_stats_easy.items():
            f.write(f"{key} = {value:.2f}\n")

        f.write("\n## Medium Difficulty (16x16, 15% mines)\n")
        for key, value in test_stats_medium.items():
            f.write(f"{key} = {value:.2f}\n")

        f.write("\n## Hard Difficulty (16x16, 20% mines)\n")
        for key, value in test_stats_hard.items():
            f.write(f"{key} = {value:.2f}\n")

        f.write("\n## AI Weights\n")
        for key, value in best_ai.weights.items():
            f.write(f"{key} = {value:.4f}\n")

    print(f"\nComprehensive test results saved to {test_results_path}")
    print("\nTraining and testing complete!")


    def plot_training_progress(self, filepath=None, show_plot=False):
        """Plot training progress metrics and save to file without blocking execution."""
        if not self.history:
            print("No training data to plot")
            return

        # Switch to a non-interactive backend to avoid Tkinter issues
        import matplotlib
        matplotlib.use('Agg')  # Use the Agg backend (non-interactive)

        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # Extract data
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        avg_fitness = [h['avg_fitness'] for h in self.history]
        deaths = [h['deaths'] for h in self.history]
        clicks = [h['clicks'] for h in self.history]
        times = [h['time'] for h in self.history]
        wins = [h['wins'] / max(1, h['games_played']) for h in self.history]  # Win rate
        grid_sizes = [f"{h['grid_size'][0]}x{h['grid_size'][1]}" for h in self.history]
        mine_percentages = [h['mine_percentage'] * 100 for h in self.history]  # Convert to percentage

        # Plot fitness
        axs[0, 0].plot(generations, best_fitness, 'b-', label='Best Fitness')
        axs[0, 0].plot(generations, avg_fitness, 'g--', label='Avg Fitness')
        axs[0, 0].set_title('Fitness Over Generations')
        axs[0, 0].set_xlabel('Generation')
        axs[0, 0].set_ylabel('Fitness Score')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot deaths and win rate
        ax1 = axs[0, 1]
        ax1.plot(generations, deaths, 'r-', label='Deaths')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Deaths', color='r')
        ax1.tick_params(axis='y', labelcolor='r')

        ax2 = ax1.twinx()
        ax2.plot(generations, [w * 100 for w in wins], 'g-', label='Win Rate %')
        ax2.set_ylabel('Win Rate %', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax1.set_title('Deaths and Win Rate')

        # Plot clicks
        axs[1, 0].plot(generations, clicks, 'm-')
        axs[1, 0].set_title('Average Clicks per Game')
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Clicks')
        axs[1, 0].grid(True)

        # Plot times
        axs[1, 1].plot(generations, times, 'c-')
        axs[1, 1].set_title('Average Time per Game')
        axs[1, 1].set_xlabel('Generation')
        axs[1, 1].set_ylabel('Time (seconds)')
        axs[1, 1].grid(True)

        # Plot grid size and mine percentage changes
        axs[2, 0].plot(generations, mine_percentages, 'y-')
        axs[2, 0].set_title('Mine Percentage')
        axs[2, 0].set_xlabel('Generation')
        axs[2, 0].set_ylabel('Mine Percentage (%)')
        axs[2, 0].grid(True)

        # Plot grid size changes (this will be more of a step function)
        # Convert grid sizes to a numeric value for plotting
        unique_grid_sizes = list(set(grid_sizes))
        unique_grid_sizes.sort()  # Sort grids by size
        grid_size_values = [unique_grid_sizes.index(size) for size in grid_sizes]

        axs[2, 1].plot(generations, grid_size_values, 'k-')
        axs[2, 1].set_title('Grid Size Progression')
        axs[2, 1].set_xlabel('Generation')
        axs[2, 1].set_yticks(range(len(unique_grid_sizes)))
        axs[2, 1].set_yticklabels(unique_grid_sizes)
        axs[2, 1].grid(True)

        plt.tight_layout()

        # Save the plot
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(EXPORT_PATH, f"minesweeper_training_plot_{timestamp}.png")

        plt.savefig(filepath)
        print(f"Training plot saved to {filepath}")

        # Close the figure to release resources
        plt.close(fig)

        return filepath
