import matplotlib.pyplot as plt
import numpy as np
from AuxFunc import AuxFunc
from GENetLib.basis_mat import bspline_mat


# Plot original functions
class FDPlot:
    def __init__(self, result):
        self.result = result

    # Plot functions for pf_bifunc
    def pf_fdplot(self):
        plot_t = np.linspace(0, 1, 1000)
        spline_mat = bspline_mat(plot_t, AuxFunc(n = self.result['sample number'], m = self.result['nknots'], x = plot_t).knots_eq(), norder = self.result['order']) 
        for i in self.result['sample cluster']:
            for j in self.result['feature cluster']:
                total_sum = np.zeros((self.result['nknots']+self.result['order']))
                count = 0
                for m in list(i):
                    for n in list(j):
                        element = self.result['Beta'][m][n] 
                        total_sum += element
                        count += 1
                mean_beta = total_sum / count
                #plt.ylim(-5, 5)
                plt.plot(spline_mat @ mean_beta)
                plt.show()

'''
from pf_bifunc import pf_bifunc
from simulation_data import pf_sim_data
pf_simdata = pf_sim_data(n = 60, T = 10, nknots = 3, order = 3, seed = 123)['data']
pf_result = pf_bifunc(pf_simdata, nknots = 3, order = 3, gamma1 = 0.023, gamma2 = 3, 
                      theta = 1, tau = 3, max_iter = 500, eps_abs = 1e-3, eps_rel = 1e-3)
FDPlot(pf_result).pf_fdplot()
'''

'''
from matplotlib.ticker import FormatStrFormatter
plt.rcParams.update({'figure.figsize': (8, 8), 'font.size': 15})
plt.ylim(-5, 5)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.plot(np.concatenate(((spline_list[0][3] @ pf_result['Beta'][18:24]), (spline_list[0][2] @ pf_result['Beta'][12:18])), axis=0))
'''