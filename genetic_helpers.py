# genetic_helpers.py

import numpy as np

def bool_to_np(board):
    return np.array(board, dtype=int)

def get_peaks(area):
    peaks = []
    for col in range(area.shape[1]):
        column = area[:, col]
        filled_indices = np.where(column == 1)[0]
        if filled_indices.size > 0:
            peak = area.shape[0] - filled_indices[-1]
            peaks.append(peak)
        else:
            peaks.append(0)
    return np.array(peaks)

def get_row_transition(area, highest_peak):
    transitions = 0
    for row in range(int(area.shape[0] - highest_peak), area.shape[0]):
        for col in range(1, area.shape[1]):
            if area[row, col] != area[row, col - 1]:
                transitions += 1
    return transitions

def get_col_transition(area, peaks):
    transitions = 0
    for col in range(area.shape[1]):
        if peaks[col] <= 1:
            continue
        for row in range(int(area.shape[0] - peaks[col]), area.shape[0] - 1):
            if area[row, col] != area[row + 1, col]:
                transitions += 1
    return transitions

def get_bumpiness(peaks):
    bumpiness = 0
    for i in range(len(peaks) - 1):
        bumpiness += abs(peaks[i] - peaks[i + 1])
    return bumpiness

def get_holes(peaks, area):
    holes = 0
    for col in range(area.shape[1]):
        if peaks[col] == 0:
            continue
        column = area[:, col]
        filled = False
        for row in range(area.shape[0]):
            if column[row] == 1:
                filled = True
            elif filled and column[row] == 0:
                holes += 1
    return holes

def get_wells(peaks):
    wells = []
    for i in range(len(peaks)):
        if i == 0:
            w = peaks[1] - peaks[0] if len(peaks) > 1 else 0
            wells.append(max(w, 0))
        elif i == len(peaks) - 1:
            w = peaks[-2] - peaks[-1] if len(peaks) > 1 else 0
            wells.append(max(w, 0))
        else:
            w1 = peaks[i - 1] - peaks[i]
            w2 = peaks[i + 1] - peaks[i]
            wells.append(max(w1, 0) if w1 >= w2 else max(w2, 0))
    return wells
