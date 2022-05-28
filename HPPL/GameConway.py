import numpy as np
from mpi4py import MPI
import imageio
import sys
import argparse


  
def change_borders(board, paddings=[1, 1, 1, 1]): # up right down left
    height, width = board.shape
    board_with_borders = np.pad(board, ((paddings[0], paddings[2]), (paddings[1], paddings[3])), 'constant', constant_values=0)
    return board_with_borders

    
def game_step(board):
    next_state = np.array(board)
    height, width = next_state.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            count_neighbors = np.sum(board[i-1:i+2, j-1 : j+2]) - board[i, j]
            is_alive = board[i, j]

            if is_alive and (count_neighbors < 2 or count_neighbors > 3):
                next_state[i, j] = 0
            if not is_alive and count_neighbors == 3:
                next_state[i, j] = 1

    return next_state


def game():
    pars = argparse.ArgumentParser()
    pars.add_argument('direction', type=str)
    pars.add_argument('n_iter', type=int)


    args = pars.parse_args()

    height = args.height 
    width = args.width
    n_iter = args.n_iter
    init_path = args.init_path
    out_path = args.out_path
    answer_path = args.answer_path



    comm = MPI.COMM_WORLD
    rank = comm.Getother_rank()
    size = comm.Get_size()
    start = np.load(init_path)
    first_state = np.zeros((height, width), dtype=np.uint8)
    first_state[:start.shape[0], :start.shape[1]] = start
    current_state = first_state[:, width // size * rank : width // size * (rank + 1) + ((width % size - 1) if rank == size - 1 else 0)]

    history = [current_state]
    prevother_rank = (rank - 1 if rank > 0 else size-1)
    nextother_rank = ((rank + 1) % size)
    
    for step_number in range(n_iter):
        if 0 < rank < size - 1:
            comm.send(current_state[:, 0], dest=rank - 1, tag=step_number)
            comm.send(current_state[:, -1], dest=rank + 1, tag=step_number)

            border_left = change_borders(current_state, [0, 0, 0, 1])
            border_left[:, 0] = comm.recv(source=rank - 1, tag=step_number)

            border_right = change_borders(border_left, [0, 1, 0, 0])
            border_right[:, -1] = comm.recv(source=rank + 1, tag=step_number)
            
            current_state = game_step(change_borders(border_right, [1, 0, 1, 0]))
            
            history.append(current_state)
        
        if rank == size - 1 and rank > 0:
            comm.send(current_state[:, 0], dest=rank - 1, tag=step_number)

            border_left = change_borders(current_state, [0, 0, 0, 1])
            border_left[:, 0] = comm.recv(source=rank - 1, tag=step_number)
            
            current_state = game_step(change_borders(border_left, [1, 1, 1, 0]))
            
            history.append(current_state)
            
        if rank == 0 and size > 1:
            comm.send(current_state[:, -1],  dest=1, tag=step_number)

            border_right = change_borders(current_state, [0, 1, 0, 0])
            border_right[:, -1] = comm.recv(source=1, tag=step_number)
            
            current_state = game_step(change_borders(border_right, [1, 0, 1, 1]))
            
            history.append(current_state)
            
        if rank == 0 and size == 1:
            current_state = game_step(change_borders(current_state))
            history.append(current_state)

    all_history = comm.gather(history, root=0)

    if rank == 0:
        frames = []
        cells_number = np.zeros(n_iter)

        for step in range(n_iter):
            frame = np.zeros((height, width), dtype=np.uint8)
            for other_rank in range(size):
                frame[:, width // size * other_rank:width // size * (other_rank + 1) + ((width % size - 1) if other_rank == size - 1 else 0)] = all_history[other_rank][game_step]

            cells_number[step] = frame.sum()
            frames.append(frame * 255)

        imageio.mimsave(out_path, frames)
        np.save(answer_path, cells_number)
        
if __name__ == '__main__':
    game()
