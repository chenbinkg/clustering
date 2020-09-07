# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:57:46 2020

@author: chenbin
"""
import pickle
import pandas as pd
import numpy as np
import time
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType
import pyspark.sql.functions as F
import mu_esda_kemclustering.mysql_db_handler as mysql

class LabelData(object):
    #MODEL_DIR = 'file:///mapa_f10/mu_esda_kemclustering/model/'
    #MODEL_DIR = 'file:///home/hdfsf15/pg1_esda/mu_esda_kemclustering/model/'
    @classmethod
    def label_failure_modes(cls, site, did, rd_item, df, model_dir, sc):
        '''
        : param site: site, e.g. 'fab15', 'fab10'
        : did: design id, e.g. 'Z32D'
        : rd_item: rd bin in string format, e.g. 'rdC'
        '''
        start_time = time.time()
        #Convert to Pandas dataframe
#        df = df.toPandas()
#        if 'FBD_REGION' in df.columns:
#            df['FBD_REGION'] = df['FBD_REGION'].apply(lambda x : cls.label_zone(x))
        
        labelled_failure_modes = []
        df = df.withColumn("row_id", F.monotonically_increasing_id())
        model_features_list, model_name_list, model_dir_list = cls.__read_model_name(site, did, rd_item, model_dir)
        print(model_dir_list)
        if len(model_name_list)>0:
            for name, features, dirname in zip(model_name_list, model_features_list, model_dir_list):
                features_missing = [e for e in features if e not in df.columns]
                if len(features_missing)>0:
                    print('Features %s missing for model %s' %(','.join(features_missing), name))
                else:
                    print(dirname)
                    print(features)
                    try:
                        model = RandomForestClassificationModel.load(str(dirname))
                        # model = LinearSVCModel.load(model_dir)
                        assembler = VectorAssembler(inputCols=features, outputCol="features")
                        # Set maxCategories so features with > 4 distinct values are treated as continuous.
                        newData = assembler.transform(df)
                        df_i = model.transform(newData)
                        #df_i = cls.pred_rf_model_spark(dirname, feature, name, df)
                        df_i = df_i.withColumnRenamed("prediction", name)
                        df = df.join(df_i.select("row_id", name), ("row_id"))   
                        labelled_failure_modes.append(name)
                        print('Labelling done for: ', name)
                    except:
                        print('Labelling failed for: ', name)
            if len(labelled_failure_modes)>0:
                df = df.withColumn('total', sum(df[col] for col in labelled_failure_modes))
                df_labelled = df.filter(df.total > 0)
                df_unlabelled = df.filter(df.total == 0)
            else:
                df_labelled = []
                df_unlabelled = df
        else:
            df_labelled = []
            df_unlabelled = df
            print('No models found for: %s, %s, %s' %(site, did, rd_item))
        
        print('Labelling time = ', time.time() - start_time)
        start_time = time.time()
        if df_labelled != []:
            df_labelled = df_labelled.toPandas()
        else:
            df_labelled = pd.DataFrame()
        df_unlabelled = df_unlabelled.toPandas()
        print('Pandas df conversion time = ', time.time() - start_time)
        return df_labelled, df_unlabelled, labelled_failure_modes

    @classmethod
    def __score_model(cls, model_dir, model_features, model_name, df):
        loaded_model = pickle.load(open(model_dir, 'rb'))
        pred = loaded_model.predict(df[model_features].values)
        pred = np.round(pred, 0)
        pred = pred.astype(int)
        df[model_name] = pred
        return df
    
    @classmethod
    def pred_rf_model_spark(cls, model_dir, feature_col, df_new):
        print('model loading start')
        # model = GBTClassificationModel.load(model_dir)
        model = RandomForestClassificationModel.load(str(model_dir))
        # model = LinearSVCModel.load(model_dir)
        assembler = VectorAssembler(inputCols=feature_col, outputCol="features")
        # Set maxCategories so features with > 4 distinct values are treated as continuous.
        newData = assembler.transform(df_new)
        predictions = model.transform(newData)
        return predictions
    
    @classmethod
    def __read_model_name(cls, site, did, probe_item, model_dir):
        sql_string = "SELECT * FROM model_catalog where FAB = '{}' \
                        and DESIGN_ID = '{}' and BIN = '{}'".format(site.replace('fab',''), 
                                                                    did, probe_item)
        query_data = mysql.DB_QUERY(sql_string)
        df_model = query_data.sqlquery()
        if len(df_model) > 0:
            df_model_g = df_model.groupby('Failure_Mode')['ECAT_Parameter'].apply(lambda x: list(x)).reset_index()
            model_name_list = df_model_g['Failure_Mode'].tolist()
            model_features_list = df_model_g['ECAT_Parameter'].tolist()
            model_dir_list = [model_dir + '{}/{}/{}/'.format(site, did, probe_item) + e for e in model_name_list]
        else:
            model_features_list, model_name_list, model_dir_list = [], [], []
        return model_features_list, model_name_list, model_dir_list

    @classmethod
    def read_model_ecat(cls, site, did, rd_item, esda_items_1, esda_items_d, extra_esda_items, model_dir):
        model_features_list, _, _ = cls.__read_model_name(site, did, rd_item, model_dir)
        features = []
        for col in model_features_list:
            features.extend(col)
        features = list(set(features))
        if 'FBD_REGION' in features:
            features.remove('FBD_REGION')
        extra_ecats = list(set(features) - set(esda_items_1+esda_items_d+extra_esda_items))
        return features, extra_ecats
    
    @classmethod
    def label_zone(cls, zone):
        #Zone encoding
        if zone == 'A' :
            return 1
        if zone == 'B' :
            return 2
        if zone == 'C':
            return 3
        if zone  == 'D':
            return 4
        if zone == 'E':
            return 5
        else:
            return 0
