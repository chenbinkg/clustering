# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:57:17 2020

@author: chenbin
"""
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PlotECATCorr(object):
    #RESULT_DIR = "/mapa_f10/mu_esda_kemclustering"
    #RESULT_DIR = "/home/hdfsf10w/pg1_esda/mu_esda_kemclustering/result"
    @classmethod
    def ecat_scatter_plot(cls, job_id, clustering_models, df_unlabelled,
                          cluster_pareto_df, cluster_cols, plot_result_dir):
        scatter_plot_dir = os.path.join(plot_result_dir, job_id)
        os.makedirs(scatter_plot_dir, mode = 0o777, exist_ok=True)
        for model in clustering_models:
            sns.set_context("paper", rc={"axes.labelsize":14, 'title_fontsize':14})
            #Choose top 5 ECATs to plot
            plot_cols = cluster_cols[0:min(len(cluster_cols), 5)]
            df_plot_grouped = df_unlabelled.groupby([model])
            df_plot = df_plot_grouped.apply(lambda x: cls.down_sampling(x)).reset_index(drop=True)
            sns_plot = sns.pairplot(df_plot[plot_cols + [model]], 
                                    hue=model, palette="husl")
            sns_plot._legend.remove()
            plt.legend(bbox_to_anchor=(1.8, 2.5), loc="best", fontsize='x-large', title_fontsize='20')
            file_name = 'scatter_plot_' + model + '.png'
            sns_plot.savefig(os.path.join(scatter_plot_dir, file_name))
            # plot_groups = df_unlabelled.groupby(model)
            # for cluster in sorted(list(plot_groups.groups)):
            #     df_pareto = cluster_pareto_df[cluster_pareto_df["cluster"] == cluster]
            #     sorted_feature = df_pareto["esda_item"].tolist()
            #     '''Pairwise x-y plot based on Most Important Features for Each Cluster'''
            #     vip_num = min(5, len(cluster_cols))
            #     vip_col = sorted_feature[0:vip_num]
            #     cluster_i_cat = "Corr_plot_" + model + "_" + cluster
            #     #            df_new[cluster_i_cat] = np.where(df_new[groupby]==i, 'MainGroup_' + str(i), 'Rest')
            #     df_unlabelled[cluster_i_cat] = np.where(df_unlabelled[model]==cluster, "Within Cluster", "Outside Cluster")
            #     cls.plot_vip_to_vip_correlation(vip_col, df_unlabelled, cluster_i_cat, scatter_plot_dir)
            #     df_unlabelled.drop(cluster_i_cat, axis=1)
    
    @classmethod
    def down_sampling(cls, x):
        sample_size = x.shape[0]
        sample_size = min(sample_size, 9000)
        #print('%s sample size: %d' %(x.name, sample_size))
        return x.sample(n=sample_size)
    
    @classmethod
    def nCr(cls, n,r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)
    
    @classmethod
    def plot_vip_to_vip_correlation(cls, vip_col, df_new, cluster_i_cat, scatter_plot_dir):
        '''plot the vip to vip correlation, no saving to pdf'''
        import itertools
        vip_num = len(vip_col)
        combination = cls.nCr(vip_num, 2)
        if combination > 6:
            columns = 2
            rows = combination/2
            x_len = 12
        else:
            columns = 1
            rows = combination
            x_len = 6
        y_len = rows*6
        vip_x_list = []
        vip_y_list = []
        vip_x_label = []
        vip_y_label = []
        for m, vip in enumerate(vip_col):
            for n in range(m+1, vip_num):
                vip_x = vip
                vip_x_list.append(vip_x)
                vip_x_label.append(vip)
                vip_y = vip_col[n]
                vip_y_list.append(vip_y)
                vip_y_label.append(vip_col[n])
        scatter_group = df_new[cluster_i_cat].unique().tolist()
        scatter_group.sort()
        colors = itertools.cycle(["b", "r"])
        fig3 = plt.figure(figsize=(x_len, y_len))
        fig3.suptitle(cluster_i_cat, fontsize=24, weight='bold', color="blue")
        outlier_filter_quantile = 0.9999
        for i in range(0, int(combination)):
            ax3 = fig3.add_subplot(rows, columns, i+1)
            # df_plot = df_new[df_new[vip_x_list[i]] < df_new[vip_x_list[i]].quantile(outlier_filter_quantile)]
            # df_plot = df_plot[df_plot[vip_y_list[i]] < df_plot[vip_y_list[i]].quantile(outlier_filter_quantile)]
            # print("Filtered ", df_plot.shape[0] - df_new.shape[0], 
            #       " points for plotting x-y correlation for x-y:", vip_x_list[i], "/", vip_y_list[i])
            for group in scatter_group:
                df_group = df_new[df_new[cluster_i_cat]==group]
                ax3.scatter(df_group[vip_x_list[i]], df_group[vip_y_list[i]], marker = 'o', 
                            c = next(colors), alpha = 0.8, s=12, label = group)
    #        df_new.plot.scatter(x=vip_x, y=vip_y, s=3,
    #                            c=cluster_i_cat, colormap='viridis',
    #                            ax=ax3)
            ax3.set_xlabel(vip_x_label[i], fontsize=22, weight='bold')
            ax3.set_ylabel(vip_y_label[i], fontsize=22, weight='bold')
            ax3.legend(loc="upper right", framealpha=0.9, frameon=True, fontsize=28, prop=dict(weight='bold', size=15))
            ax3.set_rasterized(True)    
        #                    plt.legend()
        plt.tight_layout(w_pad=2, h_pad=3)
        plt.subplots_adjust(top=0.92)
        pic_name = cluster_i_cat + '.jpg'
        pic_dir = os.path.join(scatter_plot_dir, pic_name)
        plt.savefig(pic_dir)
        #plt.show()
        plt.close()
