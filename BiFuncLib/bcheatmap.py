import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def ma_palette(low="blue", mid="white", high="red", k=50):
    if mid is None:
        colors = [low, high]
    else:
        colors = [low, mid, high]
    cmap = LinearSegmentedColormap.from_list("ma_palette", colors, N=k)
    return cmap


def bcheatmap(X, res, cexR=1.5, cexC=1.25, axisR=False, axisC=True,
              heatcols=None, clustercols=None, allrows=False, allcolumns=True):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if heatcols is None:
        heatcols = ma_palette(low="blue", mid="white", high="red", k=50)
    if clustercols is None:
        clustercols = [
            "black", "red", "blue", "green", "orange", "purple",
            "brown", "pink", "gray", "olive"]
    number = res.Number
    fig, (ax, cax) = plt.subplots(
        1, 2,
        gridspec_kw={'width_ratios': [8.5, 1.5]},
        figsize=(10, 8))
    if number == 1:
        rowmat = res.RowxNumber
        colmat = res.NumberxCol.T
        if isinstance(rowmat, pd.DataFrame):
            rowmat = rowmat.values
        if isinstance(colmat, pd.DataFrame):
            colmat = colmat.values
        roworder = np.concatenate([
            np.where(rowmat[:, 0])[0],
            np.where(~rowmat[:, 0])[0]
        ])
        colorder = np.concatenate([
            np.where(colmat[:, 0])[0],
            np.where(~colmat[:, 0])[0]
        ])
        roworder = [int(x) for x in roworder]
        colorder = [int(x) for x in colorder]
        X_ordered = X[np.ix_(roworder, colorder)]
        nr, nc = X_ordered.shape
        ax.imshow(
            np.flipud(X_ordered),
            cmap=heatcols,
            interpolation='nearest',
            aspect='auto',
            origin='lower',
            extent=(0.5, nc + 0.5, 0.5, nr + 0.5)
        )
        if axisC:
            ax.set_xticks(np.arange(1, nc + 1))
            if hasattr(res, 'NumberxCol') and hasattr(res.NumberxCol, 'columns'):
                ax.set_xticklabels(res.NumberxCol.columns[colorder],
                                   rotation=90, fontsize=cexC*10)
            else:
                ax.set_xticklabels(colorder)
        if axisR:
            ax.set_yticks(np.arange(1, nr + 1))
            if hasattr(res, 'RowxNumber') and hasattr(res.RowxNumber, 'index'):
                ax.set_yticklabels(res.RowxNumber.index[roworder],
                                   fontsize=cexR*10)
            else:
                ax.set_yticklabels(roworder)
        rin1 = np.where(np.isin(roworder, np.where(rowmat[:, 0])[0]))[0]
        cin1 = np.where(np.isin(colorder, np.where(colmat[:, 0])[0]))[0]
        effective_nr = nr if allrows else len(roworder)
        xl, yb = 0.5, effective_nr - len(rin1) + 0.5
        xr, yt = len(cin1) + 0.5, effective_nr + 0.5
        rect = patches.Rectangle(
            (xl, yb),
            xr - xl, yt - yb,
            fill=False,
            edgecolor=clustercols[0],
            linewidth=4
        )
        ax.add_patch(rect)
    else:
        rowmat = res.RowxNumber
        if isinstance(rowmat, pd.DataFrame):
            rowmat = rowmat.values
        overlap = np.sum(rowmat, axis=1)
        roworder = np.where(overlap == number)[0].tolist()
        if number > 2:
            for i in range(number - 2):
                innext = np.intersect1d(
                    np.where(rowmat[:, i])[0],
                    np.where(rowmat[:, i+1])[0]
                )
                nooverlap = np.where(
                    rowmat[:, i] & (np.sum(rowmat, axis=1) == 1)
                )[0]
                for l in range(1, number - i):
                    temp = np.intersect1d(
                        np.where(rowmat[:, i])[0],
                        np.where(rowmat[:, i+l])[0]
                    )
                    temp = np.setdiff1d(temp, innext)
                    roworder = np.unique(
                        np.concatenate((roworder, temp))
                    ).tolist()

                roworder = np.unique(
                    np.concatenate((roworder, nooverlap, innext))
                ).tolist()
        innext = np.intersect1d(
            np.where(rowmat[:, number-2])[0],
            np.where(rowmat[:, number-1])[0])
        nooverlap = np.where(
            rowmat[:, number-2] & (np.sum(rowmat, axis=1) == 1)
        )[0]
        roworder = np.unique(
            np.concatenate((roworder, nooverlap, innext))
        ).tolist()
        nooverlap = np.where(
            rowmat[:, number-1] & (np.sum(rowmat, axis=1) == 1)
        )[0]
        roworder = np.unique(
            np.concatenate((roworder, nooverlap))
        ).tolist()
        if allrows:
            extra = np.setdiff1d(np.arange(rowmat.shape[0]), roworder)
            roworder = roworder + extra.tolist()
        colmat = res.NumberxCol.T
        if isinstance(colmat, pd.DataFrame):
            colmat = colmat.values
        overlap = np.sum(colmat, axis=1)
        colorder = np.where(overlap == number)[0].tolist()
        if number > 2:
            for i in range(number - 2):
                innext = np.intersect1d(
                    np.where(colmat[:, i])[0],
                    np.where(colmat[:, i+1])[0]
                )
                nooverlap = np.where(
                    colmat[:, i] & (np.sum(colmat, axis=1) == 1)
                )[0]
                for l in range(1, number - i):
                    temp = np.intersect1d(
                        np.where(colmat[:, i])[0],
                        np.where(colmat[:, i+l])[0]
                    )
                    temp = np.setdiff1d(temp, innext)
                    colorder = np.unique(
                        np.concatenate((colorder, temp))
                    ).tolist()

                colorder = np.unique(
                    np.concatenate((colorder, nooverlap, innext))
                ).tolist()
        innext = np.intersect1d(
            np.where(colmat[:, number-2])[0],
            np.where(colmat[:, number-1])[0]
        )
        nooverlap = np.where(
            colmat[:, number-2] & (np.sum(colmat, axis=1) == 1)
        )[0]
        colorder = np.unique(
            np.concatenate((colorder, nooverlap, innext))
        ).tolist()

        nooverlap = np.where(
            colmat[:, number-1] & (np.sum(colmat, axis=1) == 1)
        )[0]
        colorder = np.unique(
            np.concatenate((colorder, nooverlap))
        ).tolist()

        if allcolumns:
            extra = np.setdiff1d(np.arange(colmat.shape[0]), colorder)
            colorder = colorder + extra.tolist()
        roworder = [int(x) for x in roworder]
        colorder = [int(x) for x in colorder]
        X_ordered = X[np.ix_(roworder, colorder)]
        nr, nc = X_ordered.shape
        ax.imshow(
            np.flipud(X_ordered),
            cmap=heatcols,
            interpolation='nearest',
            aspect='auto',
            origin='lower',
            extent=(0.5, nc + 0.5, 0.5, nr + 0.5)
        )
        if axisC:
            ax.set_xticks(np.arange(1, nc + 1))
            if hasattr(res, 'NumberxCol') and hasattr(res.NumberxCol, 'columns'):
                ax.set_xticklabels(res.NumberxCol.columns[colorder],
                                   rotation=90, fontsize=cexC*10)
            else:
                ax.set_xticklabels(colorder)
        if axisR:
            ax.set_yticks(np.arange(1, nr + 1))
            if hasattr(res, 'RowxNumber') and hasattr(res.RowxNumber, 'index'):
                ax.set_yticklabels(res.RowxNumber.index[roworder],
                                   fontsize=cexR*10)
            else:
                ax.set_yticklabels(roworder)
        rin1 = np.where(np.isin(roworder, np.where(rowmat[:, 0])[0]))[0]
        cin1 = np.where(np.isin(colorder, np.where(colmat[:, 0])[0]))[0]
        effective_nr = nr if allrows else len(roworder)
        xl, yb = 0.5, effective_nr - len(rin1) + 0.5
        xr, yt = len(cin1) + 0.5, effective_nr + 0.5
        rect = patches.Rectangle(
            (xl, yb),
            xr - xl, yt - yb,
            fill=False,
            edgecolor=clustercols[0],
            linewidth=4
        )
        ax.add_patch(rect)
        for i in range(1, number):
            rin = np.where(np.isin(roworder, np.where(rowmat[:, i])[0]))[0]
            if len(rin) == 0:
                continue
            rstart, rstop = [rin[0]], []
            for j in range(1, len(rin)):
                if rin[j] != rin[j-1] + 1:
                    rstop.append(rin[j-1])
                    rstart.append(rin[j])
            rstop.append(rin[-1])
            cin = np.where(np.isin(colorder, np.where(colmat[:, i])[0]))[0]
            if len(cin) == 0:
                continue
            cstart, cstop = [cin[0]], []
            for j in range(1, len(cin)):
                if cin[j] != cin[j-1] + 1:
                    cstop.append(cin[j-1])
                    cstart.append(cin[j])
            cstop.append(cin[-1])
            for rs, re in zip(rstart, rstop):
                for cs, ce in zip(cstart, cstop):
                    xl_rect = cs - 0.5
                    yb_rect = effective_nr - re + 0.5
                    xr_rect = ce + 0.5
                    yt_rect = effective_nr - rs + 1.5
                    rect_patch = patches.Rectangle(
                        (xl_rect, yb_rect),
                        xr_rect - xl_rect,
                        yt_rect - yb_rect,
                        fill=False,
                        edgecolor=clustercols[i],
                        linewidth=4,
                        hatch='/' * (i+1)
                    )
                    ax.add_patch(rect_patch)
    norm = mpl.colors.Normalize(
        vmin=np.min(X_ordered),
        vmax=np.max(X_ordered)
    )
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=heatcols)
    mappable.set_array([])
    plt.colorbar(
        mappable,
        cax=cax,
        orientation='vertical',
        ticks=[np.min(X_ordered), 0, np.max(X_ordered)]
    ).ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.show()

