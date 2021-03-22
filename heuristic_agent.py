def rival(obs, config):
    #imports
    import numpy as np
    import random
    #deixa caure una peca
    def drop_piece(grid, col, piece, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = piece
        return next_grid
    #ens diu si en les casellas que hem seleccionat podem fer la heuristica
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)
    #ens diu quants grups de casellas satisfan el nombre que hem dit en la heuristica
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        for row in range(config.rows):
            for col in range(config.columns):
                #horizontal
                if col + config.inarow <= config.columns:
                    window = list(grid[row, col : col + config.inarow])
                    if check_window(window, num_discs, piece, config):
                        num_windows += 1
                #vertical
                if row + config.inarow <= config.rows:
                    window = list(grid[row : row + config.inarow, col])
                    if check_window(window, num_discs, piece, config):
                        num_windows += 1
                #diagonal positiva
                if col + config.inarow <= config.columns and row + config.inarow <= config.rows:
                    window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                    if check_window(window, num_discs, piece, config):
                        num_windows += 1
                #diagonal negativa
                if col + config.inarow <= config.columns and row >= config.inarow - 1:
                    window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                    if check_window(window, num_discs, piece, config):
                        num_windows += 1
        return num_windows
    #calcula la heuristica
    def get_heuristic(grid, mark, config):
        num_fours = count_windows(grid, 4, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_twos = count_windows(grid, 2, mark, config)
        num_twos_opp = count_windows(grid, 2, mark%2+1, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        A = 10000
        B = 100
        C = 10
        D = -10
        E = -1000
        score = A*num_fours + B*num_threes + C*num_twos + D*num_twos_opp + E*num_threes_opp
        return score
    #funcio per saber si es un node final
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow
    #mira tots els estats per saber si es una leaf node
    def is_terminal_node(grid, config):
        #mirem si hi ha empat
        if list(grid[0, :]).count(0) == 0:
            return True
        for row in range(config.rows):
            for col in range(config.columns):
                #horizontal
                if col + config.inarow <= config.columns:
                    window = list(grid[row, col : col + config.inarow])
                    if is_terminal_window(window, config):
                        return True
                #vertical
                if row + config.inarow <= config.rows:
                    window = list(grid[row : row + config.inarow, col])
                    if is_terminal_window(window, config):
                        return True
                #diagonal positiva
                if col + config.inarow <= config.columns and row + config.inarow <= config.rows:
                    window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                    if is_terminal_window(window, config):
                        return True
                #diagonal negativa
                if col + config.inarow <= config.columns and row >= config.inarow - 1:
                    window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                    if is_terminal_window(window, config):
                        return True
        return False
    # Minimax implementation
    def minimax_alphabeta(node, depth, maximizingPlayer, alpha, beta, mark, config):
        is_terminal = is_terminal_node(node, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if depth == 0 or is_terminal:
            return get_heuristic(node, mark, config)
        if maximizingPlayer:
            value = -np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                value = max(value, minimax_alphabeta(child, depth-1, False, alpha, beta, mark, config))
                alpha = max(alpha, value);
                if (alpha > beta):
                    break
            return value
        else:
            value = np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1, config)
                value = min(value, minimax_alphabeta(child, depth-1, True, alpha, beta, mark, config))
                beta = min(beta, value);
                if (beta <= alpha):
                    break
            return value
    #ens diu lo bo que es un moviment
    def score_move(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        score = minimax_alphabeta(next_grid, nsteps-1, False, -1e9, 1e9, mark, config)
        return score
    #agent
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    grid = np.asarray(obs.board).reshape(config.rows, config.columns);
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, 3) for col in valid_moves]))
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    return random.choice(max_cols)
