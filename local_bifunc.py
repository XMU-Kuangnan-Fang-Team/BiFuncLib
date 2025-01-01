import pandas as pd


# Input data from R
from simulation_data import local_sim_data
filename = "C:/Users/YYBG-WANG/Desktop/科研/4.Biclustering_python/对照数据/local_sample/data.csv"
local_data = pd.read_csv(filename)
vector_list = [local_data.values[i] for i in range(local_data.values.shape[0])]
local_simdata_py = local_sim_data(n = 100, T = 100, sigma = 0.5, seed = 42)
local_simdata = {'data': vector_list,
                 'location': local_simdata_py['location'],
                 'sample cluster': local_simdata_py['sample cluster']}