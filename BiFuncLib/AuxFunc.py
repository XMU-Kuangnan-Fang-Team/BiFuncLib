import numpy as np
import pandas as pd
import itertools
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt


class AuxFunc:
    def __init__(self, n, m = None, x = None, V = None):
        self.n = n
        self.m = m if m is not None else None
        self.x = np.array(x) if x is not None else None
        self.V = np.array(V) if V is not None else None

    # Generate break sequence in spline expansion
    def knots_eq(self):
        return np.r_[self.x.min(), np.linspace(self.x.min(), self.x.max(), self.m+2)[1:-1], self.x.max()]

    # Construct the adjacency matrix
    def create_adjacency(self, plot = True):
        differences = np.linalg.norm(self.V, axis=0)
        connected_ix = np.where(differences == 0)[0]
        index = pd.DataFrame(itertools.combinations(range(self.n), 2))
        i = index.iloc[list(connected_ix),0]
        j = index.iloc[list(connected_ix),1]
        A = lil_matrix((self.n, self.n))
        A[i, j] = 1
        if plot == True:
            plt.figure()
            plt.spy(A, markersize=1)
            plt.show()
        A_array = A.toarray()
        return A_array

