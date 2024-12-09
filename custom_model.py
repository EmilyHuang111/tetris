# custom_model.py

import numpy as np
from genetic_helpers import bool_to_np
from copy import deepcopy
from piece import Piece
from board import Board

class CUSTOM_AI_MODEL:
    def __init__(self, genotype_file='data/best_genotype_epoch_15.npy', aggregate='lin'):
        """
        Initializes the AI model by loading the trained genotype.
        
        :param genotype_file: Path to the saved genotype file.
        :param aggregate: Type of aggregation function used ('lin', 'exp', etc.).
        """
        self.genotype = np.load(genotype_file)
        self.aggregate = aggregate

    def valuate(self, board, aggregate='lin'):
        """
        Computes the evaluation score based on the board state and genotype.
        """
        peaks = get_peaks(board)
        highest_peak = np.max(peaks)
        holes = get_holes(peaks, board)
        wells = get_wells(peaks)

        rating_funcs = {
            'agg_height': np.sum(peaks),
            'n_holes': np.sum(holes),
            'bumpiness': get_bumpiness(peaks),
            'num_pits': np.count_nonzero(np.count_nonzero(board, axis=0) == 0),
            'max_wells': np.max(wells),
            'n_cols_with_holes': np.count_nonzero(np.array(holes) > 0),
            'row_transitions': get_row_transition(board, highest_peak),
            'col_transitions': get_col_transition(board, peaks),
            'cleared': np.count_nonzero(np.mean(board, axis=1))
        }

        # Only linear aggregation is implemented
        if aggregate == 'lin':
            ratings = np.array([rating_funcs['agg_height'],
                                rating_funcs['n_holes'],
                                rating_funcs['bumpiness'],
                                rating_funcs['num_pits'],
                                rating_funcs['max_wells'],
                                rating_funcs['n_cols_with_holes'],
                                rating_funcs['row_transitions'],
                                rating_funcs['col_transitions'],
                                -rating_funcs['cleared']])  # Negative because clearing rows is good
            aggregate_rating = np.dot(ratings, self.genotype)
        else:
            # Implement other aggregation functions if needed
            aggregate_rating = 0

        return aggregate_rating

    def get_best_move(self, board, piece, depth=1):
        """
        Determines the best move based on the current board and piece.
        
        :param board: Current state of the board.
        :param piece: Current piece to place.
        :param depth: Search depth (not used in this simple implementation).
        :return: Tuple (best_x, best_piece).
        """
        best_x = -1
        best_piece = None
        max_value = -np.inf

        for rotation in range(4):
            rotated_piece = piece.get_next_rotation()
            for x in range(board.width):
                try:
                    y = board.drop_height(rotated_piece, x)
                except Exception:
                    continue

                board_copy = deepcopy(board.board)
                for pos in rotated_piece.body:
                    board_copy[y + pos[1]][x + pos[0]] = True

                np_board = bool_to_np(board_copy)
                value = self.valuate(np_board, aggregate=self.aggregate)

                if value > max_value:
                    max_value = value
                    best_x = x
                    best_piece = rotated_piece

        return best_x, best_piece

# Ensure you have all helper functions imported or defined within this file
from genetic_helpers import get_peaks, get_holes, get_wells, get_bumpiness, get_row_transition, get_col_transition
