import pandas as pd
import numpy as np
from pathlib import Path


# tcell data
def tcell():
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    data_path = current_dir / "real_data" / "tcell.csv"
    tcell_data = pd.read_csv(data_path)
    tcell_label = [0] * 10 + [1] * 34
    return {'data' : tcell_data, 'label' : tcell_label}


# growth data
def growth():
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    data_path = current_dir / "real_data" / "growth.csv"
    growth_data = pd.read_csv(data_path).iloc[4:-2]
    growth_label = [0] * 54 + [1] * 39
    # Reformat data
    t = growth_data.iloc[:, 0].values
    subjects = growth_data.columns[1:]
    growth_reformat = pd.DataFrame({'time': t})
    for subj in subjects:
        h = growth_data[subj].values
        n = len(h)
        v = np.zeros(n)
        v[0] = (h[1] - h[0]) / (t[1] - t[0])
        for i in range(1, n-1):
            v[i] = (h[i+1] - h[i-1]) / (t[i+1] - t[i-1])
        v[-1] = (h[-1] - h[-2]) / (t[-1] - t[-2])
        growth_reformat[subj] = v
    x = (t - t.min()) / (t.max() - t.min())
    return {'data' : growth_reformat.iloc[:, 1:],
            'location' : x,
            'label' : growth_label}