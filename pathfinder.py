import argparse
from collections import deque

STUDENT_ID = 'a1901793'
DEGREE = 'UG'

modes = ["debug", "release"]
algorithms = ["bfs", "ucs", "astar"]
heuristics = ["euclidian", "manhattan"]

def cost(curr_pos, next_pos, board):
    return 1 + max(0, int(board[next_pos[0]][next_pos[1]]) - int(board[curr_pos[0]][curr_pos[1]]))

def bfs(board_size, start, end, board):
    rows, cols = board_size
    directions = [(1,0), (-1,0), (0,-1), (0,1)]
    
    queue = deque([(start, [start], 0)])
    visited = {start: 0}
    process = []
    
    while queue:
        (x, y), path, total_cost = queue.popleft()
        process.append((x, y))
        
        if (x, y) == end:
            return path, process
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and board[nx][ny] != 'X':
                new_cost = total_cost + cost((x, y), (nx, ny), board)
                if (nx, ny) not in visited or new_cost < visited[(nx, ny)]:
                    visited[(nx, ny)] = new_cost
                    queue.append(((nx, ny), path + [(nx, ny)], new_cost))
    
    return None, None

def read_map(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        print(lines)
    
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