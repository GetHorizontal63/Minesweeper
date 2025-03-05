import pygame
import random
import time
import sys
from pygame.locals import *

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (192, 192, 192)
DARK_GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
MAROON = (128, 0, 0)
TURQUOISE = (64, 224, 208)
ORANGE = (255, 165, 0)

# Number colors
NUMBER_COLORS = {
    1: BLUE,
    2: GREEN,
    3: RED,
    4: PURPLE,
    5: MAROON,
    6: TURQUOISE,
    7: BLACK,
    8: DARK_GRAY
}

# Game parameters
TILE_SIZE = 30
BORDER = 2
TOP_MARGIN = 60  # Space for timer, mine counter, etc.

# Font setup
FONT = pygame.font.SysFont('Arial', 16)
LARGE_FONT = pygame.font.SysFont('Arial', 24)
BUTTON_FONT = pygame.font.SysFont('Arial', 18)


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)

        text_surface = BUTTON_FONT.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def is_clicked(self, pos, event):
        # Make sure we're checking a mouse button down event and the position is within the button
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False


class Minesweeper:
    def __init__(self, rows, cols, difficulty):
        self.rows = rows
        self.cols = cols
        self.difficulty = difficulty
        self.mine_percentage = self._get_mine_percentage(difficulty)
        self.num_mines = int((rows * cols) * self.mine_percentage)
        self.board = [[' ' for _ in range(cols)] for _ in range(rows)]
        self.revealed = [[False for _ in range(cols)] for _ in range(rows)]
        self.flagged = [[False for _ in range(cols)] for _ in range(rows)]
        self.first_move = True
        self.game_over = False
        self.win = False
        self.start_time = None
        self.end_time = None

        # Calculate window size
        self.width = cols * TILE_SIZE + 2 * BORDER
        self.height = rows * TILE_SIZE + 2 * BORDER + TOP_MARGIN

        # Setup display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Minesweeper - {difficulty.capitalize()}")

        # Load or create images
        self.images = {
            'flag': self._create_flag_image(),
            'mine': self._create_mine_image(),
            'wrong_flag': self._create_wrong_flag_image()
        }

    def _get_mine_percentage(self, difficulty):
        difficulty_map = {
            'easy': 0.05,  # 5% mines
            'medium': 0.10,  # 10% mines
            'hard': 0.20,  # 20% mines
            'expert': 0.40,  # 40% mines
            'insane': 0.80  # 80% mines
        }
        return difficulty_map.get(difficulty.lower(), 0.10)  # Default to medium if invalid

    def _create_flag_image(self):
        surface = pygame.Surface((TILE_SIZE - 10, TILE_SIZE - 10), pygame.SRCALPHA)
        # Flag pole
        pygame.draw.line(surface, BLACK, (10, 5), (10, 20), 2)
        # Flag
        pygame.draw.polygon(surface, RED, [(10, 5), (20, 10), (10, 15)])
        # Base
        pygame.draw.rect(surface, BLACK, (5, 18, 10, 3))
        return surface

    def _create_mine_image(self):
        surface = pygame.Surface((TILE_SIZE - 10, TILE_SIZE - 10), pygame.SRCALPHA)
        center = (10, 10)
        radius = 8

        # Main circle
        pygame.draw.circle(surface, BLACK, center, radius)

        # Spikes
        for i in range(8):
            angle = i * 45
            x1 = center[0] + radius * pygame.math.Vector2(1, 0).rotate(angle).x
            y1 = center[1] + radius * pygame.math.Vector2(1, 0).rotate(angle).y
            x2 = center[0] + (radius + 4) * pygame.math.Vector2(1, 0).rotate(angle).x
            y2 = center[1] + (radius + 4) * pygame.math.Vector2(1, 0).rotate(angle).y
            pygame.draw.line(surface, BLACK, (x1, y1), (x2, y2), 2)

        # Highlight
        pygame.draw.circle(surface, WHITE, (center[0] - 3, center[1] - 3), 2)

        return surface

    def _create_wrong_flag_image(self):
        surface = pygame.Surface((TILE_SIZE - 10, TILE_SIZE - 10), pygame.SRCALPHA)
        # Copy the flag image
        surface.blit(self._create_flag_image(), (0, 0))
        # Add an X
        pygame.draw.line(surface, RED, (5, 5), (15, 15), 2)
        pygame.draw.line(surface, RED, (15, 5), (5, 15), 2)
        return surface

    def place_mines(self, safe_row, safe_col):
        # Place mines randomly, ensuring first clicked cell is safe
        mines_placed = 0
        safe_cells = []

        # Mark the first clicked cell and its neighbors as safe
        for r in range(max(0, safe_row - 1), min(self.rows, safe_row + 2)):
            for c in range(max(0, safe_col - 1), min(self.cols, safe_col + 2)):
                safe_cells.append((r, c))

        while mines_placed < self.num_mines:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)

            if (r, c) not in safe_cells and self.board[r][c] != 'X':
                self.board[r][c] = 'X'
                mines_placed += 1

        # Calculate numbers for cells adjacent to mines
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] != 'X':
                    count = self._count_adjacent_mines(r, c)
                    self.board[r][c] = str(count) if count > 0 else ' '

    def _count_adjacent_mines(self, row, col):
        count = 0
        for r in range(max(0, row - 1), min(self.rows, row + 2)):
            for c in range(max(0, col - 1), min(self.cols, col + 2)):
                if (r, c) != (row, col) and self.board[r][c] == 'X':
                    count += 1
        return count

    def get_tile_rect(self, row, col):
        x = col * TILE_SIZE + BORDER
        y = row * TILE_SIZE + BORDER + TOP_MARGIN
        return pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

    def get_cell_from_pos(self, pos):
        x, y = pos
        if y < TOP_MARGIN:
            return None, None

        row = (y - TOP_MARGIN - BORDER) // TILE_SIZE
        col = (x - BORDER) // TILE_SIZE

        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None, None

        return row, col

    def reveal(self, row, col):
        if self.game_over or self.flagged[row][col]:
            return

        if self.first_move:
            self.start_time = time.time()
            self.place_mines(row, col)
            self.first_move = False

        # Hit a mine
        if self.board[row][col] == 'X':
            self.game_over = True
            self.end_time = time.time()
            self.reveal_all_mines()
            return

        # Already revealed
        if self.revealed[row][col]:
            return

        # Reveal this cell
        self.revealed[row][col] = True

        # If empty space, recursively reveal adjacent cells
        if self.board[row][col] == ' ':
            for r in range(max(0, row - 1), min(self.rows, row + 2)):
                for c in range(max(0, col - 1), min(self.cols, col + 2)):
                    if not self.revealed[r][c]:
                        self.reveal(r, c)

        # Check for win condition
        self.check_win()

    def toggle_flag(self, row, col):
        if self.game_over or self.revealed[row][col]:
            return

        self.flagged[row][col] = not self.flagged[row][col]
        self.check_win()

    def chord(self, row, col):
        """Reveal adjacent cells if flags match the number"""
        if not self.revealed[row][col] or self.board[row][col] == ' ' or self.board[row][col] == 'X':
            return

        # Count flagged neighbors
        flagged_count = 0
        for r in range(max(0, row - 1), min(self.rows, row + 2)):
            for c in range(max(0, col - 1), min(self.cols, col + 2)):
                if (r, c) != (row, col) and self.flagged[r][c]:
                    flagged_count += 1

        # If flagged count matches the number, reveal unflagged neighbors
        if flagged_count == int(self.board[row][col]):
            for r in range(max(0, row - 1), min(self.rows, row + 2)):
                for c in range(max(0, col - 1), min(self.cols, col + 2)):
                    if (r, c) != (row, col) and not self.flagged[r][c] and not self.revealed[r][c]:
                        self.reveal(r, c)

    def reveal_all_mines(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == 'X':
                    self.revealed[r][c] = True

    def check_win(self):
        for r in range(self.rows):
            for c in range(self.cols):
                # If there's a cell that's not a mine and not revealed, game is not won yet
                if self.board[r][c] != 'X' and not self.revealed[r][c]:
                    return

        # All non-mine cells are revealed
        self.win = True
        self.game_over = True
        self.end_time = time.time()

        # Flag all mines
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == 'X' and not self.flagged[r][c]:
                    self.flagged[r][c] = True

    def get_remaining_flags(self):
        flagged_count = sum(row.count(True) for row in self.flagged)
        return self.num_mines - flagged_count

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return int(end - self.start_time)

    def draw_tile(self, row, col):
        rect = self.get_tile_rect(row, col)

        if self.revealed[row][col]:
            # Revealed tile
            pygame.draw.rect(self.screen, WHITE, rect)
            pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)

            # Draw tile content
            if self.board[row][col] == 'X':
                # Mine
                if self.game_over and not self.win:
                    pygame.draw.rect(self.screen, RED, rect)
                self.screen.blit(self.images['mine'], (rect.x + 5, rect.y + 5))
            elif self.board[row][col] != ' ':
                # Number
                number = int(self.board[row][col])
                text = FONT.render(self.board[row][col], True, NUMBER_COLORS.get(number, BLACK))
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
        else:
            # Unrevealed tile
            pygame.draw.rect(self.screen, GRAY, rect)

            # 3D effect
            pygame.draw.line(self.screen, WHITE, rect.topleft, rect.topright, 2)
            pygame.draw.line(self.screen, WHITE, rect.topleft, rect.bottomleft, 2)
            pygame.draw.line(self.screen, DARK_GRAY, rect.bottomleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, DARK_GRAY, rect.topright, rect.bottomright, 2)

            # Flag
            if self.flagged[row][col]:
                if self.game_over and self.board[row][col] != 'X':
                    # Wrong flag
                    self.screen.blit(self.images['wrong_flag'], (rect.x + 5, rect.y + 5))
                else:
                    # Normal flag
                    self.screen.blit(self.images['flag'], (rect.x + 5, rect.y + 5))

    def draw_board(self):
        # Background
        self.screen.fill(GRAY)

        # Draw top panel
        pygame.draw.rect(self.screen, GRAY, (0, 0, self.width, TOP_MARGIN))

        # Draw mines counter
        mines_text = LARGE_FONT.render(f"Mines: {self.get_remaining_flags()}", True, BLACK)
        self.screen.blit(mines_text, (10, 10))

        # Draw timer
        timer_text = LARGE_FONT.render(f"Time: {self.get_elapsed_time()}", True, BLACK)
        timer_rect = timer_text.get_rect()
        timer_rect.topright = (self.width - 10, 10)
        self.screen.blit(timer_text, timer_rect)

        # Draw difficulty
        diff_text = FONT.render(f"Difficulty: {self.difficulty.capitalize()}", True, BLACK)
        diff_rect = diff_text.get_rect()
        diff_rect.center = (self.width // 2, 35)
        self.screen.blit(diff_text, diff_rect)

        # Draw tiles
        for row in range(self.rows):
            for col in range(self.cols):
                self.draw_tile(row, col)

        # Game over or win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = GREEN if self.win else RED

            # Semi-transparent overlay
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 150))
            self.screen.blit(overlay, (0, 0))

            # Message
            message_text = pygame.font.SysFont('Arial', 48).render(message, True, color)
            message_rect = message_text.get_rect(center=(self.width // 2, self.height // 2 - 20))

            # Time
            time_text = LARGE_FONT.render(f"Time: {self.get_elapsed_time()} seconds", True, BLACK)
            time_rect = time_text.get_rect(center=(self.width // 2, self.height // 2 + 20))

            # Draw message and time
            self.screen.blit(message_text, message_rect)
            self.screen.blit(time_text, time_rect)

            # New Game button
            pygame.draw.rect(self.screen, GRAY,
                             (self.width // 2 - 60, self.height // 2 + 50, 120, 40))
            pygame.draw.rect(self.screen, BLACK,
                             (self.width // 2 - 60, self.height // 2 + 50, 120, 40), 2)

            new_game_text = BUTTON_FONT.render("New Game", True, BLACK)
            new_game_rect = new_game_text.get_rect(
                center=(self.width // 2, self.height // 2 + 70))
            self.screen.blit(new_game_text, new_game_rect)

    def handle_click(self, pos, button):
        # Check if game is over and click is on the new game button
        if self.game_over:
            new_game_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 50, 120, 40)
            if new_game_rect.collidepoint(pos) and button == 1:
                return "new_game"
            return None

        row, col = self.get_cell_from_pos(pos)
        if row is None or col is None:
            return None

        if button == 1:  # Left click
            if self.revealed[row][col]:
                self.chord(row, col)
            else:
                self.reveal(row, col)
        elif button == 3:  # Right click
            self.toggle_flag(row, col)

        return None


def create_main_menu():
    # Increased height window
    screen = pygame.display.set_mode((400, 800))
    pygame.display.set_caption("Minesweeper")

    title_font = pygame.font.SysFont('Arial', 48)

    # Define colors for selected buttons
    SELECTED_COLOR = (150, 200, 255)  # Light blue for selected options

    # Add more space between size and difficulty sections
    # Move size buttons up a bit
    size_buttons = [
        Button(130, 120, 140, 40, "Small (8x8)", GRAY, WHITE),
        Button(130, 170, 140, 40, "Medium (16x16)", GRAY, WHITE),
        Button(130, 220, 140, 40, "Wide (16x30)", GRAY, WHITE),
        Button(130, 270, 140, 40, "Large (24x30)", GRAY, WHITE),
    ]

    # Move difficulty section down to create more space
    difficulty_buttons = [
        Button(130, 380, 140, 40, "Easy (5%)", GRAY, WHITE),
        Button(130, 430, 140, 40, "Medium (10%)", GRAY, WHITE),
        Button(130, 480, 140, 40, "Hard (20%)", GRAY, WHITE),
        Button(130, 530, 140, 40, "Expert (40%)", GRAY, WHITE),
        Button(130, 580, 140, 40, "Insane (80%)", GRAY, WHITE),
    ]

    # Move the start button to match the new layout
    start_button = Button(150, 650, 100, 50, "Start", GREEN, (100, 255, 100))

    selected_size_index = 0  # Default Small
    selected_difficulty_index = 0  # Default Easy
    selected_size = (8, 8)
    selected_difficulty = "easy"

    running = True
    while running:
        # Clear screen
        screen.fill(WHITE)

        # Handle events
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                print(f"Mouse button {event.button} pressed at {event.pos}")

                # Check if any size button was clicked
                for i, button in enumerate(size_buttons):
                    if button.is_clicked(event.pos, event):
                        print(f"Size button {i} clicked")
                        selected_size_index = i
                        if i == 0:
                            selected_size = (8, 8)
                        elif i == 1:
                            selected_size = (16, 16)
                        elif i == 2:
                            selected_size = (16, 30)
                        elif i == 3:
                            selected_size = (24, 30)

                # Check if any difficulty button was clicked
                for i, button in enumerate(difficulty_buttons):
                    if button.is_clicked(event.pos, event):
                        print(f"Difficulty button {i} clicked")
                        selected_difficulty_index = i
                        if i == 0:
                            selected_difficulty = "easy"
                        elif i == 1:
                            selected_difficulty = "medium"
                        elif i == 2:
                            selected_difficulty = "hard"
                        elif i == 3:
                            selected_difficulty = "expert"
                        elif i == 4:
                            selected_difficulty = "insane"

                # Check if start button was clicked
                if start_button.is_clicked(event.pos, event):
                    print("Start button clicked")
                    return selected_size, selected_difficulty

        # Update hover states for all buttons
        for button in size_buttons + difficulty_buttons + [start_button]:
            button.check_hover(mouse_pos)

        # Draw title
        title_text = title_font.render("Minesweeper", True, BLACK)
        title_rect = title_text.get_rect(center=(200, 50))
        screen.blit(title_text, title_rect)

        # Draw size selection - CENTERED
        size_label = LARGE_FONT.render("Grid Size:", True, BLACK)
        size_label_rect = size_label.get_rect(center=(200, 90))
        screen.blit(size_label, size_label_rect)

        # Draw size buttons with selected button highlighted
        for i, button in enumerate(size_buttons):
            # Save original color
            original_color = button.color

            # Change color if this is the selected button
            if i == selected_size_index:
                button.color = SELECTED_COLOR

            # Draw the button
            button.draw(screen)

            # Restore original color for hover effects to work properly
            button.color = original_color

        # Draw difficulty selection - CENTERED and with more space
        difficulty_label = LARGE_FONT.render("Difficulty:", True, BLACK)
        difficulty_label_rect = difficulty_label.get_rect(center=(200, 350))
        screen.blit(difficulty_label, difficulty_label_rect)

        # Draw difficulty buttons with selected button highlighted
        for i, button in enumerate(difficulty_buttons):
            # Save original color
            original_color = button.color

            # Change color if this is the selected button
            if i == selected_difficulty_index:
                button.color = SELECTED_COLOR

            # Draw the button
            button.draw(screen)

            # Restore original color for hover effects to work properly
            button.color = original_color

        # Draw start button
        start_button.draw(screen)

        # Update display
        pygame.display.flip()

    return (16, 16), "medium"  # Default if window is closed

def main():
    while True:
        # Show main menu
        size, difficulty = create_main_menu()

        # Create game
        game = Minesweeper(size[0], size[1], difficulty)

        # Game loop
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    action = game.handle_click(event.pos, event.button)
                    if action == "new_game":
                        running = False

            # Draw board
            game.draw_board()
            pygame.display.flip()

            # Cap framerate
            clock.tick(30)


if __name__ == "__main__":
    main()
