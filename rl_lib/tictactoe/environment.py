import numpy as np
import random
from collections import deque

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.move_count = 0  # Add move counter
        
    def reset(self, starting_player):
        self.board = np.zeros((3, 3))
        self.current_player = starting_player
        self.move_count = 0  # Reset move counter
        return self.get_state()
    
    def get_state(self):
        """Convert board to a flat array for the neural network"""
        return self.board.flatten()
    
    def get_valid_moves(self):
        """Return list of empty positions"""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]
    
    def make_move(self, position):
        """Make a move and return (next_state, reward, done)"""
        i, j = position
        
        # Check if move is valid
        if self.board[i][j] != 0:
            return self.get_state(), -10, True
        
        # Make move
        self.board[i][j] = self.current_player
        self.move_count += 1
        
        # Calculate reward scaling factor (decreases as moves increase)
        # With this formula, reward at move 1 is 2.0, at move 5 is 1.0, at move 9 is 0.2
        reward_scale = 2.0 * (10 - self.move_count) / 9
        
        # Check for winner
        winner = self._check_winner()
        if winner == self.current_player:
            return self.get_state(), reward_scale * 5, True  # Increased base reward and scaled
        elif winner == -self.current_player:
            return self.get_state(), reward_scale * -5, True  # Increased base reward and scaled
        elif len(self.get_valid_moves()) == 0:
            return self.get_state(), 0, True
        
        # Switch player
        self.current_player *= -1
        return self.get_state(), 0, False
    
    def _check_winner(self):
        """Check if there's a winner"""
        # Check rows
        for i in range(3):
            if abs(sum(self.board[i])) == 3:
                return self.board[i][0]
        
        # Check columns
        for i in range(3):
            if abs(sum(self.board[:, i])) == 3:
                return self.board[0][i]
        
        # Check diagonals
        if abs(self.board[0][0] + self.board[1][1] + self.board[2][2]) == 3:
            return self.board[1][1]
        if abs(self.board[0][2] + self.board[1][1] + self.board[2][0]) == 3:
            return self.board[1][1]
        
        return 0


def display_board(board):
    """Display the tic tac toe board"""
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    print('-------------')
    for i in range(3):
        row = '|'
        for j in range(3):
            row += f' {symbols[board[i][j]]} |'
        print(row)
        print('-------------')

def play_demonstration_game(agent1, agent2, env):
    """Play and display a full game between two trained agents"""
    state = env.reset(starting_player = 1)
    done = False
    print("\nDemonstration game between trained agents:")
    print("Initial board:")
    display_board(env.board)
    
    while not done:
        # Player 1's turn
        valid_moves = env.get_valid_moves()
        action1 = agent1.select_action(state, valid_moves)
        next_state, reward, done = env.make_move(action1)
        print(f"\nPlayer 1 (X) moves to position {action1}")
        display_board(env.board)
        
        if not done:
            # Player 2's turn
            valid_moves = env.get_valid_moves()
            action2 = agent2.select_action(next_state, valid_moves)
            state, reward, done = env.make_move(action2)
            print(f"\nPlayer 2 (O) moves to position {action2}")
            display_board(env.board)
        else:
            if reward >0:
                print("Player 1 (X) wins!")
            elif reward <0:
                print("Player 2 (O) wins!")
            else:
                print("It's a draw!")
            break