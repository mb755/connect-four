import numpy as np

from dataclasses import dataclass


def create_board():
    return np.zeros((6, 7), dtype=int)


@dataclass(frozen=True)
class Result:
    pass


@dataclass(frozen=True)
class Success(Result):
    pass


@dataclass(frozen=True)
class Failure(Result):
    reason: str


def add_token(board, player, col):
    if board[0, col] != 0:
        return Failure(f"Column {col} is already full")
    for i in range(5, -1, -1):
        if board[i, col] == 0:
            board[i, col] = player
            return Success()
    # should never happen
    assert False
