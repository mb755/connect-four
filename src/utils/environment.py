import numpy as np


class ConnectFourEnv:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = None
        self.current_player = None
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.winner = None
        return self.get_state()

    def step(self, action):
        if self.winner is not None:
            return self.get_state(), 0, True, {}

        if not self.is_valid_action(action):
            return self.get_state(), -10, True, {"invalid_move": True}

        row = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        done = self.check_winner(row, action)
        reward = 0

        if done:
            if self.winner == self.current_player:
                reward = 1
            elif self.winner == 0:  # Draw
                reward = 0
            else:
                reward = -1

        self.current_player = (
            3 - self.current_player
        )  # Switch player (1 -> 2 or 2 -> 1)

        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.array(
            [
                (self.board == 0).astype(int),
                (self.board == 1).astype(int),
                (self.board == 2).astype(int),
            ]
        )

    def is_valid_action(self, action):
        return 0 <= action < self.cols and self.board[0][action] == 0

    def get_next_open_row(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                return r

    def check_winner(self, row, col):
        player = self.board[row][col]

        # Check horizontal
        for c in range(max(0, col - 3), min(col + 1, self.cols - 3)):
            if np.all(self.board[row, c : c + 4] == player):
                self.winner = player
                return True

        # Check vertical
        if row <= 2:
            if np.all(self.board[row : row + 4, col] == player):
                self.winner = player
                return True

        # Check diagonal (positive slope)
        for r, c in zip(range(row - 3, row + 1), range(col - 3, col + 1)):
            if 0 <= r and r + 3 < self.rows and 0 <= c and c + 3 < self.cols:
                if np.all(self.board[r : r + 4, c : c + 4].diagonal() == player):
                    self.winner = player
                    return True

        # Check diagonal (negative slope)
        for r, c in zip(range(row + 3, row - 1, -1), range(col - 3, col + 1)):
            if 0 <= r - 3 and r < self.rows and 0 <= c and c + 3 < self.cols:
                if np.all(
                    np.diagonal(np.fliplr(self.board[r - 3 : r + 1, c : c + 4]))
                    == player
                ):
                    self.winner = player
                    return True

        # Check for draw
        if np.all(self.board != 0):
            self.winner = 0  # 0 represents a draw
            return True

        return False

    def render(self):
        print(" 0 1 2 3 4 5 6")
        print("---------------")
        for row in self.board:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print(" ", end="|")
                elif cell == 1:
                    print("X", end="|")
                else:
                    print("O", end="|")
            print()
        print("---------------")
