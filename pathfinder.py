import argparse
from collections import deque
import heapq

STUDENT_ID = 'a1901793'
DEGREE = 'UG'

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

modes = ["debug", "release"]
algorithms = ["bfs", "ucs", "astar"]
heuristics = ["euclidian", "manhattan"]

def cost(curr_pos, next_pos, board):
    return 1 + max(0, int(board[next_pos[0]][next_pos[1]]) - int(board[curr_pos[0]][curr_pos[1]]))

def manhattan(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

def euclidian(start, end):
    return (start[0] - end[0])**2 + (start[1] - end[1])**2

def astar(board_size, start, end, board, heuristic):
    rows, cols = board_size
    
    queue = [(0 + heuristic(start, end), 0, "", start, [start])]
    visited = {start: 0}
    process = []
    
    while queue:
        total_f, total_g, moves, (x, y), path = heapq.heappop(queue)
        process.append((x, y))
        
        if (x, y) == end:
            return path, process
        
        for idx, (dx, dy) in enumerate(DIRS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and board[nx][ny] != 'X':
                new_g = total_g + cost((x, y), (nx, ny), board)
                if (nx, ny) not in visited or new_g < visited[(nx, ny)]:
                    visited[(nx, ny)] = new_g
                    new_f = new_g + heuristic((nx, ny), end)
                    heapq.heappush(queue, (new_f, new_g, moves+str(idx), (nx, ny), path + [(nx, ny)]))
    
    return None, None

def bfs(board_size, start, end, board):
    rows, cols = board_size
    
    queue = deque([(start, [start], 0)])
    visited = {start: 0}
    process = []
    
    while queue:
        (x, y), path, total_cost = queue.popleft()
        process.append((x, y))
        
        if (x, y) == end:
            return path, process
        
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and board[nx][ny] != 'X':
                new_cost = total_cost + cost((x, y), (nx, ny), board)
                if (nx, ny) not in visited or new_cost < visited[(nx, ny)]:
                    visited[(nx, ny)] = new_cost
                    queue.append(((nx, ny), path + [(nx, ny)], new_cost))
    
    return None, None


def ucs(board_size, start, end, board):
    rows, cols = board_size
    
    queue = [(0, "", start, [start])]
    visited = {start: 0}
    process = []
    
    while queue:
        total_cost, moves, (x, y), path = heapq.heappop(queue)
        process.append((x, y))
        
        if (x, y) == end:
            return path, process
        
        for idx, (dx, dy) in enumerate(DIRS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and board[nx][ny] != 'X':
                new_cost = total_cost + cost((x, y), (nx, ny), board)
                if (nx, ny) not in visited or new_cost < visited[(nx, ny)]:
                    visited[(nx, ny)] = new_cost
                    heapq.heappush(queue, (new_cost, moves + str(idx), (nx, ny), path + [(nx, ny)]))
    
    return None, None
    

def read_map(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    board_size = tuple(map(int, lines[0].split()))
    start_pos = tuple(map(lambda x: int(x) - 1, lines[1].split()))
    end_pos = tuple(map(lambda x: int(x) - 1, lines[2].split()))
    board = [line.split() for line in lines[3:]]
    
    return board_size, start_pos, end_pos, board

def print_board(board):
    for i, row in enumerate(board):
        for j, ele in enumerate(row):
            if isinstance(ele, int):
                board[i][j] = str(board[i][j])
        print(' '.join(row))

def print_output(mode, path, process, board):
    if mode == "debug":
        print("path:")
        if not path:
            print("null")
        else:
            ans_board = board
            for pos in path:
                if ans_board != 'X':
                    ans_board[pos[0]][pos[1]] = '*'
            print_board(ans_board)
        print("#visits:")
        if not path:
            print("...")
        else:
            vis_board = board
            for i, row in enumerate(vis_board):
                for j, ele in enumerate(row):
                    if ele != 'X':
                        vis_board[i][j] = '.'
            for pos in process:
                if vis_board[pos[0]][pos[1]] == '.':
                    vis_board[pos[0]][pos[1]] = 1
                elif isinstance(vis_board[pos[0]][pos[1]], int):
                    vis_board[pos[0]][pos[1]] += 1
            print_board(vis_board)
        print("first visit:")
        if not path:
            print("...")
        else:
            fvi_board = board
            for i, row in enumerate(fvi_board):
                for j, ele in enumerate(row):
                    if ele != 'X':
                        fvi_board[i][j] = '.'
            for i, pos in enumerate(process):
                if fvi_board[pos[0]][pos[1]] == '.':
                    fvi_board[pos[0]][pos[1]] = i+1
            print_board(fvi_board)
        print("last visit:")
        if not path:
            print("...")
        else:
            lvi_board = board
            for i, row in enumerate(lvi_board):
                for j, ele in enumerate(row):
                    if ele != 'X':
                        lvi_board[i][j] = '.'
            for i, pos in enumerate(process):
                lvi_board[pos[0]][pos[1]] = i+1
            print_board(lvi_board)

    elif mode == "release":
        if not path:
            print("null")
        else:
            ans_board = board
            for pos in path:
                if ans_board != 'X':
                    ans_board[pos[0]][pos[1]] = '*'
            print_board(ans_board)

def path_finder(mode, map, algorithm, heuristic):
    board_size, start_pos, end_pos, board = read_map(map)

    if algorithm == "bfs":
        path, process = bfs(board_size, start_pos, end_pos, board)
    elif algorithm == "ucs":
        path, process = ucs(board_size, start_pos, end_pos, board)
    elif algorithm == "astar":
        if heuristic == "manhattan":
            path, process = astar(board_size, start_pos, end_pos, board, manhattan)
        elif heuristic == "euclidian":
            path, process = astar(board_size, start_pos, end_pos, board, euclidian)
        else:
            print("Heuristic not implemented")
    else:
        print("Algorithm not implemented yet.")
    
    print_output(mode, path, process, board)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=modes)
    parser.add_argument("map", type=str)
    parser.add_argument("algorithm", choices=algorithms)
    parser.add_argument("heuristic", nargs='?', choices=heuristics)
    
    args = parser.parse_args()
    
    path_finder(args.mode, args.map, args.algorithm, args.heuristic)

# 7 2 4 X 3 3 1 1 1 6
# 2 X 2 3 1 1 3 7 2 2
# X 2 1 3 1 5 X X X X
# 4 4 * * * * * * * *
# X 3 * X 10 3 7 2 5 1
# 7 1 * 2 2 1 5 2 1 1
# * * * 3 3 5 4 4 5 4
# * X 1 1 1 1 1 2 4 3
# * 4 X X 3 3 5 4 4 1
# 3 6 2 X X 1 3 1 3 7

# Your Release Mode Output:
# 7 2 4 X 3 3 1 1 1 6
# 2 X 2 3 1 1 3 7 2 2
# X 2 1 3 1 5 X X X X
# 4 4 * * * * * * * *
# X * * X 10 3 7 2 5 1
# 7 * 3 2 2 1 5 2 1 1
# * * 1 3 3 5 4 4 5 4
# * X 1 1 1 1 1 2 4 3
# * 4 X X 3 3 5 4 4 1
# 3 6 2 X X 1 3 1 3 7