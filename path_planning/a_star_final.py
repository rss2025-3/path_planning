#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:21:56 2025

@author: blammers
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, position, g=0, h=0):
        self.position = position
        self.g = g
        self.h = h
    def f(self):
        return self.g + self.h
    def __lt__(self, other):  # For heapq
        return self.f() < other.f()

def euclidean(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def get_neighbors(pos, grid, diagonal=True):
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    if diagonal:
        directions += [(-1,-1), (-1,1), (1,-1), (1,1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = pos[0]+dx, pos[1]+dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] == 0:
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def astar(grid, start, goal, diagonal=True):
    open_list = []
    heapq.heappush(open_list, (0, Node(start, g=0, h=euclidean(start, goal))))
    
    came_from = {}
    cost_so_far = {start: 0}
    
    expanded = 0
    max_q = 1

    while open_list:
        if len(open_list) > max_q:
            max_q = len(open_list)

        _, current_node = heapq.heappop(open_list)
        current_pos = current_node.position

        if current_pos == goal:
            return reconstruct_path(came_from, current_pos), expanded, max_q
        
        expanded += 1

        for neighbor in get_neighbors(current_pos, grid, diagonal):
            new_cost = cost_so_far[current_pos] + euclidean(current_pos, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + euclidean(neighbor, goal)
                heapq.heappush(open_list, (priority, Node(neighbor, g=new_cost, h=euclidean(neighbor, goal))))
                came_from[neighbor] = current_pos

    return None, expanded, max_q

def downscale_binary(arr, block_size=10):
    new_shape = (arr.shape[0] // block_size, arr.shape[1] // block_size)
    downscaled = np.zeros(new_shape, dtype=int)
    for i in range(0, arr.shape[0], block_size):
        for j in range(0, arr.shape[1], block_size):
            block = arr[i:i+block_size, j:j+block_size]
            downscaled[i//block_size, j//block_size] = np.mean(block) > 0
    return downscaled

def downscale_coord(coord, block_size):
    return (int(coord[0]) // block_size, int(coord[1]) // block_size)

def upscale_path(path, block_size):
    scaled_path = [(i * block_size + block_size // 2, j * block_size + block_size // 2) for i, j in path]
    
    return scaled_path

def plot_path(grid, path, start, goal, filename='path_plot.png'):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys', origin='upper')
    if path:
        px, py = zip(*path)
        ax.plot(py, px, color='red')
    ax.plot(start[1], start[0], 'go')  # Start
    ax.plot(goal[1], goal[0], 'bo')    # Goal
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def a_star_final(grid, start, goal, block_size=10):
    """

    Parameters
    ----------
    grid : np.array
        2D Occupancy grid with 1s as obstacles and 0s as free space
    start : tuple
        (x,y)
    goal : tuple
        (x,y)
    block_size : int, optional
        Factor to downscale occupancy grid to improve A* runtime. The default is 10.

    Returns
    -------
    list of (x,y) tuples representing the path from start to goal.

    """
    grid = ~grid
    start = downscale_coord(start, block_size)
    goal = downscale_coord(goal, block_size)
    grid_array = downscale_binary(grid.T, block_size)
    path, expanded, max_q = astar(grid_array, start, goal)
    converted_path = None
    plot_path(grid_array, path, start, goal, filename='path_plot.png')
    if path != None:
        converted_path = upscale_path(path, block_size)
    
    return converted_path
    
