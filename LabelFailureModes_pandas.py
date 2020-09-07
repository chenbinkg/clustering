# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:57:46 2020

@author: chenbin
"""
import configparser
import os
import pickle
import numpy as np
import lightgbm


class LabelData(object):
    MODEL_CATALOG = 'model_dir.ini'
    @classmethod
    def label_failure_modes(cls, site, did, rd_item, df):
        '''
        : param site: site, e.g. 'fab15', 'fab10'
        : did: design id, e.g. 'Z32D'
        : rd_item: rd bin in string format, e.g. 'rdC'
        '''
        if site == 'fab15':
            cls.MODEL_DIR = '/home/hdfsf15/pg1_esda/mu_esda_kemclustering/model/'
        else:
            cls.MODEL_DIR = ''
        labelled_failure_modes = []
        if len(cls.MODEL_DIR)>0:
            model_features_list, model_name_list, model_dir_list = cls._read_model_name(site, did, rd_item, 
                                                                                        cls.MODEL_DIR, cls.MODEL_CATALOG)
            if len(model_name_list)>0:
                for name, features, dirname in zip(model_name_list, model_features_list, model_dir_list):
                    features_missing = [e for e in features if e not in df.columns]
                    if len(features_missing)>0:
                        print('Features %s missing for model %s' %(','.join(features_missing), name))
                    else:
                        try:
                            df = cls._score_model(dirname, features, name, df)
                            labelled_failure_modes.append(name)
                            print('Labelling done for: ', name)
                        except:
                            print('Labelling failed for: ', name)
            else:
                print('No models found for: %s, %s, %s' %(site, did, rd_item))
        else:
            print('No model catalog found for: %s' %(site))
        df_labelled = df[df[labelled_failure_modes].sum(axis=1)>0]
        df_unlabelled = df[df[labelled_failure_modes].sum(axis=1)<=0]
        return df_labelled, df_unlabelled, labelled_failure_modes

    @classmethod
    def _score_model(cls, model_dir, model_features, model_name, df):
        loaded_model = pickle.load(open(model_dir, 'rb'))
        pred = loaded_model.predict(df[model_features].values)
        pred = np.round(pred, 0)
        pred = pred.astype(int)
        df[model_name] = pred
        return df

    @classmethod
    def _read_model_name(cls, site, did, rd_item, model_dir, model_catalog):
        '''
        : param site: site, e.g. 'fab15', 'fab10'
        : did: design id, e.g. 'Z32D'
        : rd_item: rd bin in string format, e.g. 'rdC'
        '''
        model_config_file_name = os.path.join(model_dir, model_catalog)
        model_config = configparser.ConfigParser()
        model_config.read(model_config_file_name)
        section_name = "_".join([site, did, rd_item])
        feature_key = '_features'
        if model_config.has_section(section_name):
            section_dic = dict(model_config[section_name])
            model_features_list = [model_config[section_name][e].split(',') 
                                    for e in section_dic.keys() if e.endswith(feature_key)]
            model_name_list = [e.split(feature_key)[0] for e in section_dic.keys() if e.endswith(feature_key)]
            model_dir_list = [model_config[section_name][e] for e in model_name_list]
        else:
            model_features_list, model_name_list, model_dir_list = [], [], []
        return model_features_list, model_name_list, model_dir_list
