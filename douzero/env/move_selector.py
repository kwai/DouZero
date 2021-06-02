# return all moves that can beat rivals, moves and rival_move should be same type
import collections

def common_handle(moves, rival_move):
    new_moves = list()
    for move in moves:
        if move[0] > rival_move[0]:
            new_moves.append(move)
    return new_moves

def filter_type_1_single(moves, rival_move):
    return common_handle(moves, rival_move)


def filter_type_2_pair(moves, rival_move):
    return common_handle(moves, rival_move)


def filter_type_3_triple(moves, rival_move):
    return common_handle(moves, rival_move)


def filter_type_4_bomb(moves, rival_move):
    return common_handle(moves, rival_move)

# No need to filter for type_5_king_bomb

def filter_type_6_3_1(moves, rival_move):
    rival_move.sort()
    rival_rank = rival_move[1]
    new_moves = list()
    for move in moves:
        move.sort()
        my_rank = move[1]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_7_3_2(moves, rival_move):
    rival_move.sort()
    rival_rank = rival_move[2]
    new_moves = list()
    for move in moves:
        move.sort()
        my_rank = move[2]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_8_serial_single(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_9_serial_pair(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_10_serial_triple(moves, rival_move):
    return common_handle(moves, rival_move)

def filter_type_11_serial_3_1(moves, rival_move):
    rival = collections.Counter(rival_move)
    rival_rank = max([k for k, v in rival.items() if v == 3])
    new_moves = list()
    for move in moves:
        mymove = collections.Counter(move)
        my_rank = max([k for k, v in mymove.items() if v == 3])
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_12_serial_3_2(moves, rival_move):
    rival = collections.Counter(rival_move)
    rival_rank = max([k for k, v in rival.items() if v == 3])
    new_moves = list()
    for move in moves:
        mymove = collections.Counter(move)
        my_rank = max([k for k, v in mymove.items() if v == 3])
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_13_4_2(moves, rival_move):
    rival_move.sort()
    rival_rank = rival_move[2]
    new_moves = list()
    for move in moves:
        move.sort()
        my_rank = move[2]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_14_4_22(moves, rival_move):
    rival = collections.Counter(rival_move)
    rival_rank = my_rank = 0
    for k, v in rival.items():
        if v == 4:
            rival_rank = k
    new_moves = list()
    for move in moves:
        mymove = collections.Counter(move)
        for k, v in mymove.items():
            if v == 4:
                my_rank = k
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves
