import numpy as np


class BiclustResult:
    def __init__(self, params, RowxNumber, NumberxCol, Number, info):
        self.params = params
        self.RowxNumber = RowxNumber
        self.NumberxCol = NumberxCol.T
        self.Number = Number
        self.info = info
        self.cluster_row_sizes = np.sum(RowxNumber, axis=0)
        self.cluster_col_sizes = np.sum(NumberxCol, axis=0)