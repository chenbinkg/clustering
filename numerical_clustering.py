# -*- coding: utf-8 -*-
"""
Functions for kmeans/em clustering
"""
import numpy as np
import pandas as pd
import time
import scipy
import math
from math import*
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, BisectingKMeans
from kneed import DataGenerator, KneeLocator
from sklearn.mixture import GaussianMixture
# #For image processing
# import cv2
# #Image pre-trained model
# from keras.preprocessing import image
# from keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import preprocess_input
# from keras.models import Model
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

class KEMClustering(object):
    KNEE_LOCATOR_N_ITERATIONS = 1
    @classmethod
    def run_mv_clustering(cls, df, cluster_col, did_key, probe_item_list, sc, probe_item, bin_step):
        n_iterations=cls.KNEE_LOCATOR_N_ITERATIONS
        if bin_step == "Fail_bin": #"Fail_bin condition"
            df['BIN_BIT'] = np.where(df['BIN_BIT'].str.startswith(probe_item + ":"), 1, 0)
        elif (bin_step != "Fail_bin") and (bin_step is not None): #"Other bin type conditions"
            df[bin_step] = np.where((df[bin_step] == probe_item), 1, 0)
        stacked_item_list = probe_item_list + cluster_col
        cluster_models, stacked_items = [], []
        feature_col, df_norm_s, df_s, xnorm_s,\
             df_norm_t, df_t, xnorm_t = cls.__filter_transform_data(df, cluster_col, probe_item_list)
        #Find cluster number using elbow method by kmeans using spark df
        n_components = cls.__find_cluster_split_kmeans_sparkdf(feature_col, df_norm_s, n_iterations, 'kmeans', sc)
#         n_components_g = cls.__optimal_k_kmeans_gmm_spark(feature_col, df_norm_t, n_iterations, 'gmm', sc)
#         n_components_k = cls.__optimal_k_kmeans_gmm_spark(feature_col, df_norm_s, n_iterations, 'kmeans', sc)
        #Clustering using kmeans and gmm
        label_name = 'kmeans'
        df, cluster_label_kmeans = cls.__clustering_light(df_t, n_components, 'kmeans', xnorm_s, feature_col, label_name, sc)
        stacked_items.append(stacked_item_list)
        cluster_models.append(cluster_label_kmeans)
#         label_name = 'model_2'
        df, cluster_label_gmm_l1 = cls.__clustering_light(df, n_components, 'gmm', xnorm_t, feature_col, "gmm_L1", sc)
#         stacked_items.append(stacked_item_list)
#         cluster_models.append(cluster_label_gmm_l1)
        label_name = 'gmm'
        df, cluster_label_gmm_l2 = cls.__level_2_gmm_split_merge(df, cluster_label_gmm_l1, cluster_col, label_name)
        stacked_items.append(stacked_item_list)
        cluster_models.append(cluster_label_gmm_l2)
#         label_name = 'model_4'
#         df, cluster_tl_kmeans = cls.__img_transfer_learn_kmeans(df, cluster_col, label_name)
#         stacked_items.append(stacked_item_list)
#         cluster_models.append(cluster_tl_kmeans)

        # #Find cluster number using elbow method by bisecting kmeans using spark df
        # k_clusters = cls._find_cluster_split_kmeans_sparkdf(feature_col, df_norm, n_iterations, 'bisecting_kmeans', sc)
        # #Clustering using bisecting kmeans
        # df, cluster_models = cls._clustering_light(df, k_clusters, cluster_models, 'bisecting_kmeans', xnorm, feature_col, sc)
        #df[cluster_models] = df[cluster_models].astype(int)
        #df = pd.concat([df, df_outlier], sort=False)
        for model in cluster_models:
            df[model] = "cluster_" + df[model].astype(int).astype(str)
        sc.stop()
        return df, cluster_models, stacked_items

    @classmethod
    def __run_pca(cls, xnorm, ndimensions):
        pca_col = []
        for dim in np.arange(ndimensions):
            pca_col.append('PC' + str(dim+1))
        pca = PCA(n_components=ndimensions)
        pca.fit(xnorm)
        x_pca_array = pca.transform(xnorm)
        print('completed PCA with %d components' %ndimensions)
        print(pca.explained_variance_ratio_)
        print("Total PCA explained variation", pca.explained_variance_ratio_.sum())
        return x_pca_array, pca_col
    
    @classmethod
    def __run_pca_by_pct(cls, xnorm, pct):
        pca_col = []
        pca = PCA(pct, whiten=True)
        x_pca_array = pca.fit_transform(xnorm)
        for dim in np.arange(x_pca_array.shape[1]):
            pca_col.append('PC' + str(dim+1))
        print('PCA reduced dimensionality to %d from %d' %(x_pca_array.shape[1], xnorm.shape[1]))
        print("Total PCA explained variation: ", pct)
        return x_pca_array, pca_col

    @classmethod
    def __clustering_light(cls, df, n_components, cluster_method, xnorm, feature_col, label_name, sc):
        #light clustering, 1 level split by either KMeans, gmm or bisecting Kmeans
        if cluster_method == 'kmeans':
            df, cluster_label, cluster_center = cls.__k_mean(df, n_components, xnorm, label_name)
        elif cluster_method == 'gmm':
            df, cluster_label, gmm_means, gmm_covariances, gmm_centers = cls.__gmm(df, n_components, xnorm, 'L1', label_name)
        elif cluster_method == 'bisecting_kmeans':
            bisecting_kmeans_label = cls.__bisecting_k_mean(n_components, xnorm, feature_col, sc)
            df[label_name] = bisecting_kmeans_label
            cluster_label = label_name
        #TODO - perform l2_gmm and merging by esda-to-esda correlation
        #TODO - perform l2_gmm and merging by esda-to-esda correlation, esda pareto signature, and rd/esda wafer map signature
        return df, cluster_label

    @classmethod
    def __k_mean(cls, df, k_clusters, xnorm, label_name):
        from sklearn.cluster import KMeans
        kmeans_cat = label_name
        if xnorm.shape[1] > 0:
            #n_clusters = elbow point
            start_time = time.time()
            estimator = KMeans(n_clusters=k_clusters)
            estimator.fit(xnorm)
            y_pred = estimator.predict(xnorm)
            res=estimator.__dict__
            #print(res['cluster_centers_'])
            cluster_center = res['cluster_centers_']
            print("training time: ", time.time()-start_time, "(sec)")
        else:
            #no split
            y_pred = 0
            cluster_center = []
            print("no kmeans split")
        df[kmeans_cat] = y_pred + 1
        return df, kmeans_cat, cluster_center

    @classmethod
    #Bisecting Kmeans
    def __bisecting_k_mean(cls, k_clusters, xnorm, feature_col, sc):
        #k_clusters = elbow point
        start_time = time.time()
        #convert to spark df
        sqlContext = SQLContext(sc)
        df_norm = pd.DataFrame(data = xnorm, columns = feature_col)
        spark_df = sqlContext.createDataFrame(df_norm)
        #assemble vector
        vecAssembler = VectorAssembler(inputCols=feature_col, outputCol="features")
        spark_df_clustering = vecAssembler.transform(spark_df).select('features')
        bkm = BisectingKMeans().setK(k_clusters).setSeed(1).setFeaturesCol("features")
        model = bkm.fit(spark_df_clustering)
        prediction = model.transform(spark_df_clustering).select('prediction').collect()
        labels = [p.prediction for p in prediction]
        return labels

    @classmethod
    def __gmm(cls, df, n_clusters, xnorm, cluster_level_name, label_name):
        """GMM Clustering with Defined Number of Components
        """
        start_time = time.time()
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        gmm.fit(xnorm)
        gmm_clusters = gmm.predict(xnorm)
#         gmm_clusters_prob = gmm.predict_proba(xnorm)
        gmm_cat = label_name
        df[gmm_cat] = gmm_clusters + 1
#         cluster_prob_col = ['gmm_'+str(n_clusters) + cluster_level_name +'_prob_' + str(e) for e in range(0, n_clusters)]
#         df_gmm_cluster_prob = pd.DataFrame(data = gmm_clusters_prob,
#                                            columns = cluster_prob_col)
#         df = df.join(df_gmm_cluster_prob)
        gmm_means = gmm.means_
        gmm_covariances = gmm.covariances_
        print("training time: ", time.time()-start_time, "(sec)")
        gmm_centers = np.empty(shape=(gmm.n_components, xnorm.shape[1]))
        for i in range(gmm.n_components):
            try:
                density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(xnorm)
                gmm_centers[i, :] = xnorm[np.argmax(density)]
            except:
                gmm_centers[i, :] = [0]*(xnorm.shape[1])
        return df, gmm_cat, gmm_means, gmm_covariances, gmm_centers

    @classmethod
    def __find_cluster_split_kmeans_sparkdf(cls, feature_col, df_norm, n_iterations, kmeans_method, sc):
        from pyspark.ml.clustering import KMeans
        start_time = time.time()
        #convert to spark df
        sqlContext = SQLContext(sc)
        spark_df = sqlContext.createDataFrame(df_norm)
        #assemble vector
        vecAssembler = VectorAssembler(inputCols=feature_col, outputCol="features")
        spark_df_clustering = vecAssembler.transform(spark_df).select('features')
        n_components_list = []
        n_range = np.arange(2, 20)
        for iteration in np.arange(n_iterations):
            cost = []
            for k in n_range:
                if kmeans_method == 'kmeans':
                    print("Kmeans Elbow Method K = ", k)
                    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
                    model = kmeans.fit(spark_df_clustering)
                elif kmeans_method == 'bisecting_kmeans':
                    print("Bisecting Kmeans Elbow Method K = ", k)
                    bkm = BisectingKMeans().setK(k).setSeed(1).setFeaturesCol("features")
                    model = bkm.fit(spark_df_clustering)
                cost.append(model.computeCost(spark_df_clustering)) # requires Spark 2.0 or later
            print('Cluster List: ', n_range)
            print('Within Set Sum of Squared Errors: ', cost)
            n_split_knee = cls.__knee_locator(n_range, cost, 'convex', 'decreasing', 'sum_of_square_error')
            print("Recommended no. of components by knee locator: " + str(n_split_knee))
            n_components_list.append(n_split_knee)
        n_components = int(np.median(n_components_list).round(0))
        print('Recommended median number of splits: ', n_components)
        print("elbow method time: ", time.time()-start_time, "(sec)")
        return n_components
    
    @classmethod
    def __lot_wafer_outlier_filter(cls, df, probe_item_list):
        #Filter out massive map by lot and wafer
        df_outlier_all = pd.DataFrame()
        df['ooc_label']='Normal'
        for probe_item in probe_item_list:
            if probe_item == 'BIN_BIT':
                continue
#             rdv_lot_mean = df.groupby(['LOT_ID'])[probe_item].mean().sort_values(ascending=False)
#             outlier_lotlist = rdv_lot_mean[rdv_lot_mean>(rdv_lot_mean.median()+ 3*rdv_lot_mean.std())]
#             outlier_lotlist = outlier_lotlist.index.tolist()
#             df['ooc_label'] = np.where(df['LOT_ID'].isin(outlier_lotlist), 'outlier_map', 'Normal')
            rdv_wfr_mean = df[df['ooc_label']=='Normal'].groupby(['lot_wafer'])[probe_item].mean().sort_values(ascending=False)
            outlier_lotwfrlist = rdv_wfr_mean[rdv_wfr_mean>(rdv_wfr_mean.median()+ 3*rdv_wfr_mean.std())]
            outlier_lotwfrlist = outlier_lotwfrlist.index.tolist()
            df['ooc_label'] = np.where(df['lot_wafer'].isin(outlier_lotwfrlist), 'outlier_map', df['ooc_label'])
            df_outlier = df[df['ooc_label']=='outlier_map']
            df_outlier_all = df_outlier_all.append(df_outlier)
            df = df[df['ooc_label']=='Normal']
            return df, df_outlier_all

    @classmethod
    def __filter_transform_data(cls, df, cluster_col, probe_item_list, pca_flag=True, n_pca_components=6):
        #filter off extreme outlier using quantile
        #transform data using cbrt transformation, standard scaling or PCA
        #outlier filter
        #df, df_outlier = cls.__lot_wafer_outlier_filter(df, probe_item_list)
        for col in cluster_col:
            cls.filter_thresh = 0.99999
            if df[col].quantile(cls.filter_thresh) > df[col].median():
                df = df[df[col] <= df[col].quantile(cls.filter_thresh)]
        df = df.reset_index(drop=True)
        #transformation and scaling, or PCA transform
        xnorm_t = df[cluster_col].applymap(np.cbrt).values
        xnorm_s = StandardScaler().fit_transform(df[cluster_col].values)
        cluster_col_norm = [e + '_norm' for e in cluster_col]
        if len(cluster_col) <6:
            n_pca_components = len(cluster_col)
        if pca_flag:
            #scaling
            df_norm_s = pd.DataFrame(data=xnorm_s, columns = cluster_col_norm)
            xnorm_s, pca_col = cls.__run_pca(xnorm_s, n_pca_components)
            df_pca_s = pd.DataFrame(data=xnorm_s, columns = pca_col)
            df_norm_s = pd.concat([df_norm_s, df_pca_s], axis=1)
            #non scaling
            df_norm_t = pd.DataFrame(data=xnorm_t, columns = cluster_col_norm)
            xnorm_t, pca_col = cls.__run_pca(xnorm_t, n_pca_components)
            df_pca_t = pd.DataFrame(data=xnorm_t, columns = pca_col)
            df_norm_t = pd.concat([df_norm_t, df_pca_t], axis=1)
            feature_col = pca_col
        else:
            #scaling
            df_norm_s = pd.DataFrame(data=xnorm_s, columns = cluster_col_norm)
            #non scaling
            df_norm_t = pd.DataFrame(data=xnorm_t, columns = cluster_col_norm)
            feature_col = cluster_col_norm
        df_s = pd.concat([df, df_norm_s], axis=1)
        df_t = pd.concat([df, df_norm_t], axis=1)
        return feature_col, df_norm_s, df_s, xnorm_s, df_norm_t, df_t, xnorm_t
    
    @classmethod
    def __knee_locator(cls, K, distortions, curve_shape, direction, method_str):
        kneedle = KneeLocator(K, distortions, curve=curve_shape, direction=direction)
        n_clusters = kneedle.knee
        if n_clusters == None:
            n_clusters = 1
            print("hard assign to 1 for %s method" %(method_str))
        print("optimal number = %d clusters by %s method: " %(n_clusters, method_str))
        return n_clusters

    @classmethod
    def __elbow_method(cls, Xnorm, group_range):
        '''Determine Optimal Number of Clustering Using Elbow Method'''
        # k means determine k
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans
        distortions, Sum_of_squared_distances = [], []
        K, K1 = group_range, []
        max_split = Xnorm.shape[0]
        for k in K:
            if k <= max_split:
                K1.append(k)
                print("Elbow Method K = ", k)
                kmeanModel = KMeans(n_clusters=k).fit(Xnorm)
                kmeanModel.fit(Xnorm)
                distortions.append(sum(np.min(cdist(Xnorm, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/Xnorm.shape[0])
                Sum_of_squared_distances.append(kmeanModel.inertia_)
        print('Cluster List: ', K1)
        print('distortions(cdist method): ', distortions)
        print('Sum of Squared Distance: ', Sum_of_squared_distances)
        n_clusters = cls.__knee_locator(K1, distortions, 'convex', 'decreasing', 'distortion')
        n_clusters_ssd = cls.__knee_locator(K1, Sum_of_squared_distances, 'convex', 'decreasing', 'SSD')
        n_clusters = math.ceil(np.median([n_clusters, n_clusters_ssd]))
        print('Final number of clusters to split: %d' %(n_clusters))
        return n_clusters
    
    @classmethod
    def __pairwise_regression_lambda(cls, cluster_col, df, groupby_col):
        slope_array = []
        r_sqr_array = []
        sub_group_list = []
        for group in df[groupby_col].unique():
            df1 = df[df[groupby_col] == group][cluster_col]
            sub_group_list.append(group)
            print("inter-corr for group name: ", group)
            slope_all = []
            rsq_all = []
            for i, e in enumerate(cluster_col):
                corr_x = df1.apply(lambda x: stats.linregress(df1[e], x), 
                                   result_type='expand').rename(index={0: 'slope', 
                                                                       1: 'intercept', 
                                                                       2: 'rvalue', 
                                                                       3: 'p-value', 
                                                                       4:'stderr'})
                #print("inter-corr generated for ", e)
                slope_x = corr_x.loc['slope', :].tolist()
                slope_all.extend(slope_x[i+1:])
                rsq_x = (corr_x.loc['rvalue', :]**2).tolist()
                rsq_all.extend(rsq_x[i+1:])
    
            slope_array.append(slope_all)
            r_sqr_array.append(rsq_all)
        return sub_group_list, np.array(slope_array), np.array(r_sqr_array)

    @classmethod
    def __find_gmm_optimal_k(cls, Xnorm, component_range):
        '''Finding Optimal Number of GMM Clusters Using BIC method'''
        from sklearn import mixture
        GMM = mixture.GaussianMixture
        n_components = component_range
        BIC = np.zeros(n_components.shape)
        AIC = np.zeros(n_components.shape)
        if Xnorm.shape[0] <= n_components.max():
            K_opt = 1
        else:
            for i, n in enumerate(n_components):
                clf = GMM(n_components=n,
                        covariance_type='full')
                clf.fit(Xnorm)
                AIC[i] = clf.aic(Xnorm)
                BIC[i] = clf.bic(Xnorm)
                print("BIC Method for GMM, K = ", n)
            print('Cluster List: ', n_components)
            print('BIC: ', BIC)
            from scipy.signal import argrelextrema
            minm = argrelextrema(BIC, np.less)[0]
            print('minima locations: ', minm)
            knee_1 = cls.__knee_locator(n_components, BIC, 'convex', 'decreasing', 'gmm_BIC')
            print("Recommended no. of components by knee locator: " + str(knee_1))
            if len(minm) == 0:
                print("0 component returned, try knee locator method...")
                K_opt = knee_1
                if K_opt == None or K_opt == 0:
                    K_opt = 1
                    print("0 component returned for knee method, do not split...")
            else:
                K_opt = n_components[int(np.min(minm))]
                print("Min value for BIC minima: ", K_opt)
                K_opt = int(np.median((K_opt, knee_1)))
                print('Choose median (minima, knee) as the optimal number of components for GMM: ', K_opt)
        
        print("optimal number of components: ", K_opt)
        return K_opt

    @classmethod
    def __find_feature_impt_by_pareto(cls, cluster, cluster_col, col_lotid, col_waferid):
        '''Finding pareto By Calculating the Median'''
        #print(cluster.dtypes)
        df_pareto = cluster.groupby([col_lotid, col_waferid])[cluster_col].sum().median()
        data = df_pareto.values
        labels = df_pareto.index.tolist()
        sorted_importances, sorted_features = zip(*sorted(zip(data, labels), reverse=True))
        sorted_importance = list(sorted_importances)
        sorted_feature = list(sorted_features)
        return sorted_importance, sorted_feature

    @classmethod
    def __level_2_gmm_split_merge(cls, df, level_1_gmm_name, cluster_col, label_name):
        '''split by gmm on each cluster and merge using x-y regression slopes'''
        start_time = time.time()
        cls.col_lotid, cls.col_waferid, cls.col_dieX, cls.col_dieY = 'LOT_ID', 'WAFER_ID', 'DieX', 'DieY'
        df_subgroup = pd.DataFrame(data=None, columns=[cls.col_lotid, cls.col_waferid, 
                                                    cls.col_dieX, cls.col_dieY, 'Subgroup'])
        for i, cluster in df.groupby(level_1_gmm_name):
            sorted_importance, sorted_feature = cls.__find_feature_impt_by_pareto(cluster, cluster_col, 
                                                                                cls.col_lotid, cls.col_waferid)
            '''Get top 5 most impt variables for subset clustering'''
            vip_num = 5
            vip_col = sorted_feature[0:vip_num]
            '''Regrouping based on important feature identified'''
            Xnorm_vip_col = [e+'_norm' for e in vip_col]
            Xnorm_vip_i = cluster[Xnorm_vip_col].values
            Xnorm_vip_i_sample = cluster.sample(frac=0.1)[Xnorm_vip_col].values
            n_components_vip_i = cls.__find_gmm_optimal_k(Xnorm_vip_i_sample, np.arange(1, 10))
            if n_components_vip_i > 1:
                cluster, gmm_cat_i, gmm_means_i, gmm_covariances_i, gmm_centers_i = cls.__gmm(cluster, n_components_vip_i, 
                                                                                            Xnorm_vip_i, 'L2', 'gmm_L2')
            else:
                cluster['gmm_L2'] = 1
                gmm_cat_i = 'gmm_L2'
            cluster['Subgroup'] = cluster[level_1_gmm_name].astype(str) + "-" + cluster[gmm_cat_i].astype(str)
            cluster_new = cluster[[cls.col_lotid, cls.col_waferid, cls.col_dieX, cls.col_dieY, 'Subgroup']]
            df_subgroup = df_subgroup.append(cluster_new)
        df_subgroup[cls.col_dieX] = df_subgroup[cls.col_dieX].astype(int)
        df_subgroup[cls.col_dieY] = df_subgroup[cls.col_dieY].astype(int)
        df = pd.merge(df, df_subgroup, how = 'left', on=[cls.col_lotid, cls.col_waferid, cls.col_dieX, cls.col_dieY])
        df['Subgroup_value'] = df['Subgroup'].astype(str).apply(lambda x: x.replace('-', '.')).astype(float)
        df = df.sort_values('Subgroup_value')
        print(df[df['Subgroup_value'].isnull()].shape[0])
        df = df.dropna(subset=['Subgroup_value'])

        # centroid_labels = df['Subgroup'].unique().tolist()
        print(df['Subgroup'].unique())
        if len(df['Subgroup'].unique()) == len(df[level_1_gmm_name].unique()):
            #no further merging is needed if no subsplit
            df[label_name] = df[level_1_gmm_name].copy()
        else:
            sub_group_list, slope_array, r_sqr_array = cls.__pairwise_regression_lambda(cluster_col, df, "Subgroup")
            slope_array_filled_na = np.where(np.isnan(slope_array), 0, slope_array)
            df_slope_array = pd.DataFrame(data = slope_array_filled_na)
            df_slope_array.index = sub_group_list
            df_r_sqr_array = pd.DataFrame(data = r_sqr_array)
            df_r_sqr_array.index = sub_group_list

            r_sqr_array_filled_na = np.where(np.isnan(r_sqr_array), 0, r_sqr_array)
            slope_r_sqr_multipler = np.multiply(slope_array_filled_na, r_sqr_array_filled_na)
            df_slope_r_sqr_multipler = pd.DataFrame(data = slope_r_sqr_multipler)
            df_slope_r_sqr_multipler.index = sub_group_list

            '''Determine optimal number of clusters/components'''
            print('slope matrix shape: ', slope_r_sqr_multipler.shape)
            if slope_r_sqr_multipler.shape[1] == 0:
                n_clusters = 1
                print('hard assigned final split to 1.')
            else:
                n_clusters = cls.__elbow_method(slope_r_sqr_multipler, np.arange(1, 20))

            '''Run Kmeans clustering for penalized slope matrix'''
            df_slope_r_sqr_multipler, KMeans_cat, cluster_center = cls.__k_mean(df_slope_r_sqr_multipler, 
                                                                                n_clusters, slope_r_sqr_multipler,
                                                                                label_name)
            df_groupid = df_slope_r_sqr_multipler.loc[:, KMeans_cat]
            df_groupid = df_groupid.reset_index()
            df_groupid.rename(columns = {'index': 'Subgroup'}, inplace=True)
            df = pd.merge(df, df_groupid, how='left', on = ['Subgroup'])
        cluster_label = label_name
        print('level 2 gmm labels: ', df[label_name].unique())
        print("2nd level gmm training time: ", time.time()-start_time, "(sec)")
        return df, cluster_label

    @classmethod
    def __get_image_by_wafer_lot(cls, img_size_x, img_size_y, df, lot_id, wafer_id, 
                                 col_coordx, col_coordy, col_y, col_lot, col_wafer):
        img = np.zeros((img_size_x, img_size_y))
        df_1 = df[(df[col_lot].astype(str)==lot_id) & (df[col_wafer].astype(str)==wafer_id)]
        coordinate = df_1[[col_coordx, col_coordy]].values
        value = df_1[[col_y]].values
        for x in range(len(value)):
            img[coordinate[x, 0], coordinate[x, 1]] = value[x]
        return img

    @classmethod
    def __normalize_img_matrix(cls, img, v_min, v_max):
        # a colormap and a normalization instance
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=v_min, vmax=v_max)
        # map the normalized data to colors
        # image is now RGBA (512x512x4) 
        img = cmap(norm(img))
        #img = Image.fromarray(norm(img), 'RGB')
        img_32 = img.astype('float32')
        img = cv2.cvtColor(img_32, cv2.COLOR_RGBA2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    @classmethod
    def __histogram_equalization(cls, img):
        #img = cv2.cvtColor(norm1, cv2.COLOR_RGBA2RGB)
        #img_32 = img.astype('float32')
        img = img*255
        img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = img_gray.astype(np.uint8)
        equ_img = cv2.equalizeHist(img_gray)
        img_hist_equalized = cv2.cvtColor(equ_img, cv2.COLOR_GRAY2RGB)
        return img_hist_equalized

    @classmethod
    def __contrast_limited_adaptive_histogram_equalization(cls, img):
        #-----Converting image to LAB Color model----------------------------------- 
        lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        #Lab channels
        l, a, b = cv2.split(lab)
        #Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = l.astype(np.uint8)
        cl = clahe.apply(l)
        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)
        limg = cv2.merge((cl,a,b))
        #-----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final

    @classmethod
    def __convert_to_img_by_wafer_lot(cls, img_size_x, img_size_y, df, lot_id, wafer_id, 
                                    col_coordx, col_coordy, col_y, col_lot, col_wafer,
                                     v_min, v_max):
        img = cls.__get_image_by_wafer_lot(img_size_x, img_size_y, df, lot_id, wafer_id, 
                                            col_coordx, col_coordy, col_y, col_lot, col_wafer)
        img = cls.__normalize_img_matrix(img, v_min, v_max)
        #img_hist_equalized = contrast_limited_adaptive_histogram_equalization(img)
        return img

    @classmethod
    def __img_transfer_learn_kmeans(cls, df, cluster_col, label_name):
        ''' Bucketing Data By Wafer-level Top Pareto ESDAs '''
        df_lot_wafer = df.groupby(['LOT_ID','WAFER_ID'])[cluster_col].sum().idxmax(axis=1).reset_index()
        df_lot_wafer.columns = ['LOT_ID', 'WAFER_ID', 'TopESDA']
        ''' For each ESDA Bucket, Extract Features Using VGG19 '''
        feature_list = []
        df['X'] = df['DieX'] - df['DieX'].min()
        df['Y'] = df['DieY'] - df['DieY'].min()
        img_size_Y, img_size_X = df['Y'].unique().size, df['X'].unique().size
        df_lot_wafer['feature'] = ''
        start_time = time.time()
        # model = VGG19(weights='imagenet', include_top=False)
        base_model = VGG19(weights='imagenet', include_top=False)
        # model = ResNet50(weights='imagenet', include_top=False)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
        top_esda = cluster_col[0]
        v_min = df[top_esda].min()
        v_max = df[top_esda].quantile(0.97)
        for lot, wafer in zip(df_lot_wafer['LOT_ID'], df_lot_wafer['WAFER_ID']):
            #print('feature extraction on %s %s' %(lot, wafer))
            
    #        top_esda = df_lot_wafer[(df_lot_wafer['LOT_ID']==lot)&(df_lot_wafer['WAFER_ID']==wafer)]['TopESDA'].values[0]
            img_data = cls.__convert_to_img_by_wafer_lot(img_size_X, img_size_Y, df, lot, wafer, 
                                                        'X', 'Y', top_esda, 'LOT_ID', 'WAFER_ID',
                                                        v_min, v_max)
            img_data = cv2.resize(img_data, (224, 224))
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            feature = model.predict(img_data)
            feature_np = np.array(feature)
            feature_list.append(feature_np.flatten())
        total_time = time.time() - start_time
        print('Total execution time: %.2f(sec), executime time per img: %.2f(sec)' %(total_time, total_time/len(feature_list)))
        feature_list = [e.tolist() for e in feature_list]
        df_lot_wafer['feature'] = feature_list
        feature_list_np = np.array(df_lot_wafer['feature'].values.tolist())
        pca_array, pca_col = cls.__run_pca(feature_list_np, 10)
        n_clusters = cls.__elbow_method(pca_array, np.arange(2, 30))
        df_lot_wafer, _, _ = cls.__k_mean(df_lot_wafer, n_clusters, pca_array, label_name)
        cluster_label = label_name
        # df_lot_wafer.rename(columns={kmeans_cat: cluster_label}, inplace=True)
        df = pd.merge(df, df_lot_wafer[['LOT_ID','WAFER_ID', cluster_label]], on=['LOT_ID','WAFER_ID'], how='left')
        del df['X']
        del df['Y']
        return df, cluster_label
    
    @classmethod
    def __optimal_k_kmeans_gmm_spark(cls, feature_col, df_norm, n_iterations, kmeans_method, sc):
        from pyspark.ml.clustering import KMeans, GaussianMixture
        from pyspark.ml.evaluation import ClusteringEvaluator
        start_time = time.time()
        #convert to spark df
        sqlContext = SQLContext(sc)
        spark_df = sqlContext.createDataFrame(df_norm)
        #assemble vector
        vecAssembler = VectorAssembler(inputCols=feature_col, outputCol="features")
        spark_df_clustering = vecAssembler.transform(spark_df).select('features')
        n_components_list = []
        n_range = np.arange(2, 20)
        for iteration in np.arange(n_iterations):
            silh_val = []
            cost = []
            for k in n_range:
                if kmeans_method.lower() == 'kmeans':
                    print("Kmeans Elbow Method K = ", k)
                    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
                    model = kmeans.fit(spark_df_clustering)
                    cost.append(model.computeCost(spark_df_clustering)) # requires Spark 2.0 or later
                elif kmeans_method.lower() == 'gmm':
                    print("Gmm Elbow Method K = ", k)
                    gmm = GaussianMixture().setK(k).setSeed(1).setFeaturesCol("features")
                    model = gmm.fit(spark_df_clustering)
                #cost.append(model.computeCost(spark_df_clustering)) # requires Spark 2.0 or later
                predictions = model.transform(spark_df_clustering)
                # Evaluate clustering by computing Silhouette score
                evaluator = ClusteringEvaluator()
                silhouette = evaluator.evaluate(predictions)
                silh_val.append(silhouette)
            print('Cluster List: ', list(n_range))
            print('Silhouette score: ', silh_val)
            print('Sum of Square Distance Score: ', cost)
            n_split_silh = n_range[silh_val.index(np.max(silh_val))]
            if len(cost)>0:
                n_split_knee = cls.__knee_locator(n_range, cost, 'convex', 'decreasing', 'sum_of_square_error')
                print('Knee of sum of square distance: ', str(n_split_knee))
            else:
                n_split_knee = n_split_silh
            print("Recommended no. of components by Silhouette Score: " + str(n_split_silh))
            n_clusters = math.ceil(np.median([n_split_knee, n_split_silh]))
            n_components_list.append(n_clusters)
        n_components = int(np.median(n_components_list).round(0))
        print('Recommended median number of splits: ', n_components)
        print("training time: ", time.time()-start_time, "(sec)")
        return n_components
