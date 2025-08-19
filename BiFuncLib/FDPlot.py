import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from GENetLib.fda_func import bspline_mat
from GENetLib.fda_func import eval_basis, eval_fd
from GENetLib.fda_func import create_bspline_basis, create_fourier_basis
from GENetLib.plot_gene import plot_fd
import seaborn as sns
import warnings

from BiFuncLib.AuxFunc import AuxFunc
from BiFuncLib.cc_main_func import template_evaluation, medoid_evaluation, warping_function_plot


# Plot original functions
class FDPlot:
    def __init__(self, result):
        self.result = result

    # Plot functions in pf_bifunc
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

    # Plot classified curves functions in local_bifunc
    def local_individuals_fdplot(self):
        Beta_ini = self.result['Beta_ini']
        basis_fd = self.result['basisobj']
        cls_mem = self.result['cls_mem']
        n = Beta_ini.shape[1]
        n0 = 1000
        Times = np.linspace(0, 1, n0)
        Z1 = eval_basis(Times, basis_fd)
        y_center_hat_mat = np.dot(Z1, Beta_ini)
        unique_classes = np.unique(cls_mem)
        colors = sns.color_palette("tab10", len(unique_classes))
        color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
        linestyles = ['-', '--', ':', '-.', (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
        linestyle_map = {cls: linestyles[i % len(linestyles)] for i, cls in enumerate(unique_classes)}
        fig, ax = plt.subplots(figsize=(8, 6))
        for k in range(n):
            cls = cls_mem[k]
            color = color_map[cls]
            linestyle = linestyle_map[cls]
            ax.plot(Times, y_center_hat_mat[:, k], color=color, linestyle=linestyle)
        ax.set_xlabel('Time', fontsize=15)
        ax.set_ylabel('Value', fontsize=15)
        plt.show()

    # Plot estimated cluster mean in local_bifunc
    def local_center_fdplot(self):
        Alpha = self.result['centers']
        basis_fd = self.result['basisobj']
        n0 = 1000
        Times = np.linspace(0, 1, n0)
        Z1 = eval_basis(Times, basis_fd)
        K_hat = self.result['cls_num']
        y_center_hat_mat = np.zeros((n0, K_hat))
        for k in range(K_hat):
            y_center_hat_mat[:, k] = np.dot(Z1, Alpha[:, k])
        plt.figure(figsize=(8, 6))
        for k in range(K_hat):
            plt.plot(Times, y_center_hat_mat[:, k], label=f'Cluster {k + 1}')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('', fontsize=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Clusters', title_fontsize=12, fontsize=10)
        plt.show()

    # Plot functions in cc_bifunc
    def cc_fdplot(self, data, only_mean = False, aligned = False, warping = False):
        res = self.result
        params = res["Parameter"]
        alpha = params["alpha"][0]
        beta = params["beta"][0]
        const_alpha = params["const_alpha"][0]
        const_beta = params["const_beta"][0]
        shift_alignment = params["shift_alignment"][0]
        shift_max = params["shift_max"][0]
        max_iter = params["max_iter_align"][0]
        template_type = params["template_type"][0]
        if shift_alignment == False and (aligned or warping):
            warnings.warn("Warning: no aligned can be performed if results are without alignment")
            aligned = False
            warping = False
        if not aligned and warping:
            warnings.warn("Warning: no warping can be shown if aligned is False")
            warping = False
        if res['Number'] == 0:
            raise ValueError("Warning: no Cluster found")
        col_palette = [
            "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
            "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999",
            "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
            "#FFD92F", "#E5C494", "#B3B3B3"
        ]
        x_y = np.zeros((data.shape[0], data.shape[1]))
        if res['Number'] == 1:
            rowx = np.array(res['RowxNumber'])
            numberx = np.array(res['NumberxCol'])
            x_y = np.outer(rowx, numberx)
        elif res['Number'] > 1:
            for i in range(res['Number']):
                rowx_i = np.array(res['RowxNumber'])[:, i]
                numberx_i = np.array(res['NumberxCol'])[i, :]
                xy = np.outer(rowx_i, numberx_i)
                xy = xy * (i + 1)
                x_y = x_y + xy
        if np.sum(x_y == 0) > 0:
            col_palette = (["grey"] + col_palette)
        
        def plot_single_cluster(df, cluster_num, plot_title, overlay_df=None):
            plt.figure(figsize=(6, 4))
            cluster_color = col_palette[(int(cluster_num) - 1) % len(col_palette)]
            sub_df = df[df["cluster"] == cluster_num]
            for obs, sub_obs in sub_df.groupby("obs"):
                sub_obs = sub_obs.sort_values("variable")
                plt.plot(sub_obs["variable"], sub_obs["value"], 
                         color=cluster_color, alpha=0.6, linewidth=1, label='_nolegend_')
            if overlay_df is not None:
                overlay_sub = overlay_df[overlay_df["cluster"] == cluster_num]
                overlay_sub = overlay_sub.sort_values("variable")
                plt.plot(overlay_sub["variable"], overlay_sub["value"], 
                         color='black', linewidth=2, linestyle='--', label='Template')
            plt.title(f"{plot_title}\nCluster {cluster_num}")
            plt.xlabel("Variable")
            plt.ylabel("Value")
            plt.tight_layout()
            plt.show()
        
        if aligned:
            if col_palette[0] == "grey":
                col_palette = col_palette[1:]
            fun_aligned = pd.DataFrame()
            template = pd.DataFrame()
            template_mean = pd.DataFrame()
            warping_aligned = pd.DataFrame()
            if res['Number'] == 1:
                row_idx = np.where(np.array(res['RowxNumber']) == True)[0]
                col_idx = np.where(np.array(res['NumberxCol']) == True)[0]
                clust_cl = data[np.ix_(row_idx, col_idx, np.arange(data.shape[2]))]
                res_aligned = warping_function_plot(
                    res, clust_cl, template_type, 
                    alpha, beta, const_alpha, const_beta, 
                    shift_alignment, shift_max, max_iter)
                array_aligned = res_aligned["fun_mat_align"]
                mat_aligned = np.vstack([array_aligned[tt, :, :] for tt in range(array_aligned.shape[0])])
                fun_aligned_cl = pd.DataFrame(mat_aligned)
                fun_aligned_cl["cluster"] = 1
                temp_array = res_aligned["template"]
                temp_cl = np.vstack([temp_array[tt, :, :] for tt in range(temp_array.shape[0])])
                temp_cl_df = pd.DataFrame(temp_cl)
                template_mean_cl = pd.DataFrame([temp_cl_df.mean(axis=0)])
                temp_cl_df["cluster"] = 1
                template_mean_cl["cluster"] = 1
                fun_aligned = pd.concat([fun_aligned, fun_aligned_cl], ignore_index=True)
                template = pd.concat([template, temp_cl_df], ignore_index=True)
                template_mean = pd.concat([template_mean, template_mean_cl], ignore_index=True)
                warping_cl = pd.DataFrame(res_aligned["x_out"])
                warping_cl["cluster"] = 1
                warping_aligned = pd.concat([warping_aligned, warping_cl], ignore_index=True)
            
            elif res['Number'] > 1:
                for cl in range(res['Number']):
                    cluster_label = cl + 1
                    row_idx = np.where(np.array(res['RowxNumber'])[:, cl] == True)[0]
                    col_idx = np.where(np.array(res['NumberxCol'])[cl, :] == True)[0]
                    clust_cl = data[np.ix_(row_idx, col_idx, np.arange(data.shape[2]))]
                    res_aligned = warping_function_plot(
                        res, clust_cl, template_type, 
                        alpha, beta, const_alpha, const_beta, 
                        shift_alignment, shift_max, max_iter)
                    array_aligned = res_aligned["fun_mat_align"]
                    mat_aligned = np.vstack([array_aligned[tt, :, :] for tt in range(array_aligned.shape[0])])
                    fun_aligned_cl = pd.DataFrame(mat_aligned)
                    fun_aligned_cl["cluster"] = cluster_label
                    temp_array = res_aligned["template"]
                    temp_cl = np.vstack([temp_array[tt, :, :] for tt in range(temp_array.shape[0])])
                    temp_cl_df = pd.DataFrame(temp_cl)
                    template_mean_cl = pd.DataFrame([temp_cl_df.mean(axis=0)])
                    temp_cl_df["cluster"] = cluster_label
                    template_mean_cl["cluster"] = cluster_label
                    fun_aligned = pd.concat([fun_aligned, fun_aligned_cl], ignore_index=True)
                    template = pd.concat([template, temp_cl_df], ignore_index=True)
                    template_mean = pd.concat([template_mean, template_mean_cl], ignore_index=True)
                    warping_cl = pd.DataFrame(res_aligned["x_out"])
                    warping_cl["cluster"] = cluster_label
                    warping_aligned = pd.concat([warping_aligned, warping_cl], ignore_index=True)
            fun_aligned["obs"] = "ROW" + fun_aligned.index.astype(str)
            fun_aligned_melt = fun_aligned.melt(id_vars=["obs", "cluster"])
            fun_aligned_melt["variable"] = fun_aligned_melt["variable"].astype(str).str.replace("X", "", regex=False).astype(float)
            template["obs"] = "ROW" + template.index.astype(str)
            template_melt = template.melt(id_vars=["obs", "cluster"])
            template_melt["variable"] = template_melt["variable"].astype(str).str.replace("X", "", regex=False).astype(float)
            template_mean["obs"] = "ROW" + template_mean.index.astype(str)
            template_mean_melt = template_mean.melt(id_vars=["obs", "cluster"])
            template_mean_melt["variable"] = template_mean_melt["variable"].astype(str).str.replace("X", "", regex=False).astype(float)
            warping_aligned["obs"] = "ROW" + warping_aligned.index.astype(str)
            warping_aligned_melt = warping_aligned.melt(id_vars=["obs", "cluster"])
            warping_aligned_melt["variable"] = warping_aligned_melt["variable"].astype(str).str.replace("X", "", regex=False).astype(float)
            if only_mean and warping:
                clusters = sorted(template_melt["cluster"].unique())
                for cl in clusters:
                    if alpha == 0 and beta == 0:
                        plot_single_cluster(template_melt, cl, "Representative Functions")
                    else:
                        plot_single_cluster(template_melt, cl, "Representative Functions with Template", 
                                          overlay_df=template_mean_melt)
                    plot_single_cluster(warping_aligned_melt, cl, "Warping Functions")
            elif only_mean and (not warping):
                clusters = sorted(template_melt["cluster"].unique())
                for cl in clusters:
                    if alpha == 0 and beta == 0:
                        plot_single_cluster(template_melt, cl, "Representative Functions")
                    else:
                        plot_single_cluster(template_melt, cl, "Representative Functions with Template", 
                                          overlay_df=template_mean_melt)
            elif (not only_mean) and warping:
                clusters = sorted(fun_aligned_melt["cluster"].unique())
                for cl in clusters:
                    plot_single_cluster(fun_aligned_melt, cl, "Aligned Functions")
                    plot_single_cluster(warping_aligned_melt, cl, "Warping Functions")
            elif (not only_mean) and (not warping):
                clusters = sorted(fun_aligned_melt["cluster"].unique())
                for cl in clusters:
                    plot_single_cluster(fun_aligned_melt, cl, "Aligned Functions")

        else:
            n, m, p = data.shape
            fun_plot = np.vstack([data[j, :, :] for j in range(n)])
            df_fun = pd.DataFrame(fun_plot, columns=["X" + str(i) for i in range(1, p+1)])
            df_fun["obs"] = np.repeat(["ROW " + str(i) for i in range(1, n+1)], m)
            df_fun["var"] = np.tile(["COL " + str(j) for j in range(1, m+1)], n)
            df_fun["obj"] = df_fun["obs"] + df_fun["var"]
            data_frame = pd.melt(df_fun, id_vars=["obs", "var", "obj"],
                                 var_name="variable", value_name="value")
            data_frame = data_frame.sort_values("obj")
            if res['Number'] == 1:
                row_param = np.array(res['RowxNumber'])
                col_param = np.array(res['NumberxCol'])
                xy_matrix = np.outer(row_param, col_param)
            elif res['Number'] > 1:
                xy_matrix = np.zeros((n, m))
                for i in range(res['Number']):
                    row_i = np.array(res['RowxNumber'])[:, i]
                    col_i = np.array(res['NumberxCol'])[i, :]
                    xy = np.outer(row_i, col_i) * (i + 1)
                    xy_matrix += xy
            df_xy = pd.DataFrame(xy_matrix, columns=["COL " + str(j) for j in range(1, m+1)])
            df_xy["obs"] = ["ROW " + str(i) for i in range(1, n+1)]
            df_xy = pd.melt(df_xy, id_vars=["obs"],
                            var_name="variable", value_name="value")
            obs_order = ["ROW " + str(i) for i in range(n, 0, -1)]
            df_xy["obs"] = pd.Categorical(df_xy["obs"], categories=obs_order, ordered=True)
            df_xy["obj"] = df_xy["obs"].astype(str) + df_xy["variable"].astype(str)
            df_xy = df_xy[["obj", "value"]].rename(columns={"value": "cluster"})
            data_frame_cl = pd.merge(data_frame, df_xy, on="obj")
            data_frame_cl["variable"] = data_frame_cl["variable"].str.replace("X", "").astype(float)
            template = pd.DataFrame()
            template_mean = pd.DataFrame()
            if res['Number'] == 1:
                row_idx = np.where(np.array(res['RowxNumber']) == True)[0]
                col_idx = np.where(np.array(res['NumberxCol']) == True)[0]
                clust_cl = data[np.ix_(row_idx, col_idx, np.arange(p))]
                if template_type == "mean":
                    new_fun_cl = template_evaluation(clust_cl, alpha, beta, const_alpha, const_beta)
                elif template_type == "medoid":
                    new_fun_cl = medoid_evaluation(clust_cl, alpha, beta, const_alpha, const_beta)
                temp_cl = np.vstack([new_fun_cl[tt, :, :] for tt in range(new_fun_cl.shape[0])])
                temp_cl_df = pd.DataFrame(temp_cl)
                template_mean_cl = pd.DataFrame([temp_cl_df.mean(axis=0)])
                temp_cl_df["cluster"] = 1
                template = pd.concat([template, temp_cl_df], ignore_index=True)
                template_mean_cl["cluster"] = 1
                template_mean = pd.concat([template_mean, template_mean_cl], ignore_index=True)
            elif res['Number'] > 1:
                for cl in range(res['Number']):
                    row_idx = np.where(np.array(res['RowxNumber'])[:, cl] == True)[0]
                    col_idx = np.where(np.array(res['NumberxCol'])[cl, :] == True)[0]
                    clust_cl = data[np.ix_(row_idx, col_idx, np.arange(p))]
                    if template_type == "mean":
                        new_fun_cl = template_evaluation(clust_cl, alpha, beta, const_alpha, const_beta)
                    elif template_type == "medoid":
                        new_fun_cl = medoid_evaluation(clust_cl, alpha, beta, const_alpha, const_beta)
                    temp_cl = np.vstack([new_fun_cl[tt, :, :] for tt in range(new_fun_cl.shape[0])])
                    temp_cl_df = pd.DataFrame(temp_cl)
                    template_mean_cl = pd.DataFrame([temp_cl_df.mean(axis=0)])
                    temp_cl_df["cluster"] = cl + 1
                    template = pd.concat([template, temp_cl_df], ignore_index=True)
                    template_mean_cl["cluster"] = cl + 1
                    template_mean = pd.concat([template_mean, template_mean_cl], ignore_index=True)
            template["obs"] = "ROW" + template.index.astype(str)
            template_melt = pd.melt(template, id_vars=["obs", "cluster"],
                                    var_name="variable", value_name="value")
            template_melt["variable"] = template_melt["variable"].astype(str).str.replace("X", "").astype(float)
            template_mean["obs"] = "ROW" + template_mean.index.astype(str)
            template_mean_melt = pd.melt(template_mean, id_vars=["obs", "cluster"],
                                         var_name="variable", value_name="value")
            template_mean_melt["variable"] = template_mean_melt["variable"].astype(str).str.replace("X", "").astype(float)
            if only_mean:
                clusters = sorted(template_melt["cluster"].unique())
                for cl in clusters:
                    if alpha == 0 and beta == 0:
                        plot_single_cluster(template_melt, cl, "Representative Functions")
                    else:
                        plot_single_cluster(template_melt, cl, "Representative Functions with Template", 
                                          overlay_df=template_mean_melt)
            else:
                clusters = sorted(data_frame_cl["cluster"].unique())
                for cl in clusters:
                    if cl == 0:
                        continue
                    plot_single_cluster(data_frame_cl[data_frame_cl["cluster"] == cl], cl, "Functional Data")

    # Plot functions in fem_bifunc
    def fem_fdplot(self, data, fdobj):
        res = self.result
        for i in range(res['K']):
            idx  = np.argmax(res['P'][:, i])
            plt.figure(figsize=(8, 4))
            plt.plot(np.array(data['data'])[idx].T, linestyle='-', linewidth=2, color=f'C{i}')
            x_ticks = np.arange(5, 182, 6)
            plt.xticks(x_ticks, [data['dates'][j] for j in x_ticks], rotation=90)
            plt.ylim(0, 1)
            plt.title(f"Cluster {i+1} - {data['names'][idx]}")
            plt.tight_layout()
            plt.show()
        projected_data = np.dot(fdobj['coefs'].T, res['U'])
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(np.array(projected_data[:, 0]), np.array(projected_data[:, 1]), c=res['cls'],
                              cmap='tab10', s=50)
        plt.title("Discriminative space")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(scatter, label='Class')
        plt.grid(True)
        plt.show()

    # Plot functions in lbm_bifunc
    def lbm_fdplot(self, types = 'blocks'):
        x = self.result
        colors = ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
                  "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"]
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
        types = types.lower()
        all_types = ["means", "evolution", "blocks", "criterion", "likelihood", "proportions"]
        if types not in all_types:
            raise ValueError("types must be in " + str(all_types))
        old_params = plt.rcParams.copy()
        
        if types == "means":
            T = x['T']
            K = x['K']
            L = x['L']
            if x['basisName'] == 'spline':
                basis = create_bspline_basis((0, T), x['nbasis'])
            elif x['basisName'] == 'fourier':
                basis = create_fourier_basis((0, T), x['nbasis'])
            obj = {
                'basis': basis,
                'coefs': None,
                'fdnames': {'time': list(range(1, T+1)), 'reps': None, 'values': None}
            }
            if x['datatype'] == 0:
                for l in range(L):
                    obj['coefs'] = np.transpose(x['prms']['mu'][:, l, :])
                    plot_fd(obj)
                    plt.show()
            else:
                for i in range(x['datatype']):
                    for l in range(L):
                        obj['coefs'] = np.transpose(x['prms']['mu'][:, l, (i * x['nbasis']):(x['nbasis'] * (i + 1))])
                        plot_fd(obj)
                        plt.show()
    
        elif types == "evolution":
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(x['allPrms']['Alphas'], linestyle='-', lw=2)
            axs[0].set_xlabel('Iterations')
            axs[0].set_title(r'$\alpha$')
            axs[1].plot(x['allPrms']['Betas'], linestyle='-', lw=2)
            axs[1].set_xlabel('Iterations')
            axs[1].set_title(r'$\beta$')
            plt.show()
    
        elif types == "blocks":
            T = x['T']
            K = x['K']
            L = x['L']
            nbasis = x['nbasis']
            if x['basisName'] == 'spline':
                basis = create_bspline_basis((0, T), nbasis)
            elif x['basisName'] == 'fourier':
                basis = create_fourier_basis((0, T), nbasis)
            if x['datatype'] == 0:
                for k in range(K):
                    for l in range(L):
                        slice_mu = x['prms']['mu'][k, l, :]
                        mu_max = slice_mu
                        mu_min = slice_mu
                        obj = {
                            'basis': basis,
                            'coefs': np.column_stack([mu_min, mu_max]),
                            'fdnames': {'time': list(range(1, T+1)), 'reps': None, 'values': None}
                        }
                        plot_fd(obj, xlab='', ylab='')
            else:
                for r in range(x['datatype']):
                    for k in range(K):
                        for l in range(L):
                            start_idx = r * nbasis
                            end_idx = (r + 1) * nbasis
                            slice_mu = x['prms']['mu'][k, l, start_idx:end_idx]
                            mu_max = slice_mu
                            mu_min = slice_mu
                            obj = {
                                'basis': basis,
                                'coefs': np.column_stack([mu_min, mu_max]),
                                'fdnames': {'time': list(range(1, T+1)), 'reps': None, 'values': None}
                            }
                            plot_fd(obj, xlab='', ylab='')
            
        elif types == "likelihood":
            plt.figure()
            plt.plot(x['loglik'], marker='o', color='lightblue')
            plt.xlabel('Iterations')
            plt.ylabel('Complete log-likelihood')
            plt.show()
    
        elif types == "proportions":
            fig, axs = plt.subplots(1, 2)
            K = x['K']
            axs[0].bar(range(1, K+1), x['prms']['alpha'],
                       color=colors[:K])
            axs[0].set_title(r'$\alpha$')
            axs[0].set_xticks(range(1, K+1))
            L = x['L']
            axs[1].bar(range(1, L+1), x['prms']['beta'],
                       color=colors[:L])
            axs[1].set_title(r'$\beta$')
            axs[1].set_xticks(range(1, L+1))
            plt.show()
        plt.rcParams.update(old_params)

    # Plot functions in sas_bifunc
    def sas_fdplot(self):
        mod = self.result
        G = mod["mean_fd"]["coefs"].shape[1]
        rng = mod["mean_fd"]["basis"]["rangeval"]
        grid_eval = np.linspace(rng[0], rng[1], 500)
        eval_mu = eval_fd(list(grid_eval), mod["mean_fd"])
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for i in range(G):
            axs[0].plot(grid_eval, eval_mu[:, i], label=f"Cluster {i+1}")
        axs[0].axhline(0, color='grey', alpha=0.6)
        axs[0].set_title("Cluster means")
        axs[0].set_xlim(rng)
        x_vals = mod["mod"]["data"]["x"]
        axs[0].set_ylim(np.min(x_vals), np.max(x_vals))
        axs[0].legend(loc="upper right")
        axs[1].set_title("Classified observations")
        axs[1].set_xlim(rng)
        axs[1].set_xlabel("")
        axs[1].set_ylabel("")
        axs[1].set_ylim(np.min(x_vals), np.max(x_vals))
        unique_curves = np.unique(mod["mod"]["data"]["curve"])
        cmap = plt.get_cmap("tab10")
        for ii in unique_curves:
            inds = np.where(mod["mod"]["data"]["curve"] == ii)[0]
            t_inds = mod["mod"]["data"]["timeindex"][inds]
            x_plot = np.array(mod["mod"]["grid"])[t_inds - 1]
            y_plot = mod["mod"]["data"]["x"][inds]
            cluster = mod["clus"]['classes'][int(ii)-1]
            axs[1].plot(x_plot, y_plot, color=cmap(cluster % 10), linestyle="-")
        axs[1].legend([f"Cluster {i+1}" for i in range(G)], loc="upper right")
        plt.show()

    # Plot cv results in sas_bifunc 
    def sas_cvplot(self):
        mod = self.result
        comb_list_i = mod["comb_list"]
        CV_i = mod["CV"]
        sd_i = mod["CV_sd"]
        zeros_i = mod["zeros"]
        fig = plt.figure(constrained_layout=True, figsize=(12, 10))
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, height_ratios=[1, 1])
        ax1 = fig.add_subplot(spec[0, :])  # 第一行横跨两列
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[1, 1])
        n_candidates = comb_list_i.shape[0]
        x_vals = np.arange(1, n_candidates + 1)
        labels = []
        for ii in range(n_candidates):
            a = [f"{num:.1g}" for num in comb_list_i[ii, :]]
            labels.append(" ".join(a))
        ax1.plot(x_vals, CV_i, marker='o', markersize=3, color='red', linestyle='-')
        ax1.errorbar(x_vals, CV_i, yerr=sd_i, fmt='none', color='red', capsize=3)
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels(labels, rotation=90, fontsize=8)
        for xi, z in zip(x_vals, zeros_i):
            ax1.text(xi, np.max(CV_i)+sd_i.max()*0.1, f"{round(z*100)}", ha='center', fontsize=8)
        max_idx = np.argmax(CV_i)
        ax1.axvline(x_vals[max_idx], color='black', linestyle='--')
        ax1.axhline(np.max(CV_i), color='black', linestyle='--')
        ax1.set_ylabel("CV")
        par_arr = CV_i
        sds = sd_i
        zeros_arr = zeros_i
        comb_list = comb_list_i
        m1_val, m2_val, m3_val = mod["ms"]
        lamb_s = np.unique(comb_list[:,1])
        lamb_L = np.unique(comb_list[:,2])
        max_vec_nc = []
        sd_vec_nc = []
        zero_vec = []
        new_comb_list = np.zeros((len(lamb_s)*len(lamb_L), 3))
        kk = 0
        for jj in range(len(lamb_L)):
            for ii in range(len(lamb_s)):
                indexes = np.where((comb_list[:,1] == lamb_s[ii]) & (comb_list[:,2] == lamb_L[jj]))[0]
                par_index = par_arr[indexes]
                sd_index = sds[indexes]
                zero_index = zeros_arr[indexes]
                if par_index.size == 0:
                    continue
                max_local = np.argmax(par_index)
                lim = 0.5 * abs(np.max(par_index) - np.min(par_index)) if (m1_val * sd_index[max_local] > 0.5 * abs(np.max(par_index)-np.min(par_index))) else m1_val * sd_index[max_local]
                candidates = np.where(par_index[:max_local+1] >= par_index[max_local] - lim)[0]
                onese = candidates[0] if candidates.size > 0 else max_local
                max_vec_nc.append(par_index[onese])
                sd_vec_nc.append(sd_index[onese])
                zero_vec.append(zero_index[onese])
                new_comb_list[kk, :] = comb_list[indexes[onese], :]
                kk += 1
        max_vec_nc = np.array(max_vec_nc)
        sd_vec_nc = np.array(sd_vec_nc)
        zero_vec = np.array(zero_vec)
        x_vals2 = np.arange(1, new_comb_list.shape[0] + 1)
        labels2 = []
        for ii in range(new_comb_list.shape[0]):
            a = [f"{num:.1g}" for num in new_comb_list[ii, :]]
            labels2.append(" ".join(a))
        ax2.plot(x_vals2, max_vec_nc, marker='o', color='red', linestyle='-')
        ax2.errorbar(x_vals2, max_vec_nc, yerr=sd_vec_nc, fmt='none', color='red', capsize=3)
        ax2.set_xticks(x_vals2)
        ax2.set_xticklabels(labels2, rotation=90, fontsize=8)
        for xi, z in zip(x_vals2, zero_vec):
            ax2.text(xi, np.max(max_vec_nc)+sd_vec_nc.max()*0.1, f"{round(z*100)}", ha='center', fontsize=8)
        ax2.set_ylabel("CV fixed G")
        max_vec_s = []
        sd_vec_s = []
        zero_vec2 = []
        new_comb_list2 = np.zeros((len(lamb_L), 3))
        kk = 0
        for ii in range(len(lamb_L)):
            indexes = np.where(new_comb_list[:, 2] == lamb_L[ii])[0]
            if indexes.size == 0:
                continue
            par_index_sub = max_vec_nc[indexes]
            sd_index_sub = sd_vec_nc[indexes]
            zero_index_sub = zero_vec[indexes]
            max_idx_sub = np.argmax(par_index_sub)
            candidates = np.where(par_index_sub >= par_index_sub[max_idx_sub] - m2_val * sd_index_sub[max_idx_sub])[0]
            onese_final = candidates[-1] if candidates.size > 0 else max_idx_sub
            max_vec_s.append(par_index_sub[onese_final])
            sd_vec_s.append(sd_index_sub[onese_final])
            zero_vec2.append(zero_index_sub[onese_final])
            new_comb_list2[kk, :] = new_comb_list[indexes[onese_final], :]
            kk += 1
        max_vec_s = np.array(max_vec_s)
        sd_vec_s = np.array(sd_vec_s)
        zero_vec2 = np.array(zero_vec2)
        x_vals3 = np.arange(1, new_comb_list2.shape[0] + 1)
        labels3 = []
        for ii in range(new_comb_list2.shape[0]):
            a = [f"{num:.1g}" for num in new_comb_list2[ii, :]]
            labels3.append(" ".join(a))
        ax3.plot(x_vals3, max_vec_s, marker='o', color='red', linestyle='-')
        ax3.errorbar(x_vals3, max_vec_s, yerr=sd_vec_s, fmt='none', color='red', capsize=3)
        ax3.set_xticks(x_vals3)
        ax3.set_xticklabels(labels3, rotation=90, fontsize=8)
        for xi, z in zip(x_vals3, zero_vec2):
            ax3.text(xi, np.max(max_vec_s)+sd_vec_s.max()*0.1, f"{round(z*100)}", ha='center', fontsize=8)
        ax3.set_ylabel("CV fixed G and lambda_s")
        plt.show()

    # Plot functions in sparse_bifunc
    def sparse_fdplot(self, x, data):
        if len(self.result) == 4:
            clusters = self.result['cluster']
            w = self.result['w']
        else:
            clusters = self.result['result']['cluster']
            w = self.result['result']['w']
        for i in range(data.shape[1]):
            plt.plot(x, data.T[i, :], label=f'Cluster {clusters[i] + 1}', linewidth=1)
        plt.title('Sparse functional K-means')
        plt.show()
        plt.plot(x, w, linestyle='-', linewidth=2, label='Weighting function')
        plt.title('Weighting function')
        plt.show()


