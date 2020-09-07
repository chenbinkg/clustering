import pandas as pd
import numpy as np
import time
from pyspark.sql.functions import when, col, split, udf, lit
from pyspark.sql.types import StringType
import pyspark.sql.functions as F

class ProbeDataPreprocess(object):
    TOP_ESDA_PARETO_PCT = 0.97
    @classmethod
    def dataprep_pyspark(cls, probeDF, probe_item_list, probe_item, probe_dependent_item_list,
                           esda_items, esda_items_1, esda_items_d, 
                           extra_esda_items):
        """
        This function preprocess the raw data by filtering out the failed dies and convert the data to numerical type.
        :param probeDF: raw data in spark Dataframe
        :param probe_item_list: probe column in the dataframe, e.g. ['__rdC'], ['__rdV']
        :param probe_item: probe column from input, e.g. 'V', 'C' etc
        :param probe_dependent_item_list: rd column in the dataframe, e.g. ['__rdt'], ['__rdZ']
        :param esda_items: ESDA columns in the dataframe, e.g. ['BITR0', 'BITR1', ...]
        :param esda_items_1: ESDA columns tied to probe_item, e.g. ['BITR0_V_ALL', 'BITR1_V_ALL', ...]
        :param esda_items_d: ESDA columns tied to probe_dependent_item, e.g. ['BITR0_t_ALL', 'BITR1_t_ALL', ...]
        :param extra_esda_items: extra ESDA columns ['DBAWPA_VS_ALL, DBTT_V2_ALL']
        """
        '''split FID column into lot,wafer,dieX and dieY in a small df
            create lot_wafer column from lot and wafer
            merge back into main df'''
        if probeDF.rdd.isEmpty():
            print("raw_data is empty")
            raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "queried data is empty.")
        else:
            start_time = time.time()
#             esda_items_all = esda_items_1 + esda_items_d + extra_esda_items
#             probe_items_all = probe_item_list + probe_dependent_item_list
#             all_registers = esda_items_all + probe_items_all
#             probe_df = probeDF.select(split(probeDF.FID,":")).rdd.flatMap(
#                                                       lambda x: x).toDF(schema=['LotId', 'WaferId', 'DieX', 'DieY'])
#             probe_df = probe_df.withColumn('DieX', F.when(
#                                             F.col('DieX').startswith('N'),
#                                             F.concat(lit('-'), F.col('DieX').substr(2, 4))
#                                             ).otherwise(F.col('DieX').substr(2, 4)))
#             probe_df = probe_df.withColumn("DieX", probe_df["DieX"].cast("Integer"))
#             probe_df = probe_df.withColumn("DieY", probe_df["DieY"].cast("Integer"))
#             probe_df = probe_df.withColumn("lot_wafer", F.concat(col('LotId'), lit("_"), col('WaferId')))
#             '''create a row_id for merging and then discard'''
#             probeDF = probeDF.withColumn("row_id", F.monotonically_increasing_id())
#             probe_df = probe_df.withColumn("row_id", F.monotonically_increasing_id())
#             probeDF = probeDF.join(probe_df, ("row_id")).drop("row_id")

            df = probeDF.toPandas()
            df[['LotId', 'WaferId', 'DieX', 'DieY']] = df['FID'].str.split(':',expand=True)
            df['DieX'] = np.where(df['DieX'].str.startswith('N'), '-' + df['DieX'].str[1:], df['DieX'].str[1:])
            df['DieX'] = df['DieX'].astype(int)
            df['DieY'] = df['DieY'].astype(int)
            df['lot_wafer'] = df['LotId'].astype(str) + "_" + df['WaferId'].astype(str)
            print('Spark df to pandas df processing time = ', time.time() - start_time)
            df, cluster_col_pareto, esda_items_selected = cls.__preprocess_rawdata(df, probe_item_list, probe_item, 
                                                                                   probe_dependent_item_list,
                                                                                   esda_items, esda_items_1, esda_items_d, 
                                                                                   extra_esda_items)
            return df, cluster_col_pareto, esda_items_selected

    @classmethod
    def __label_zone(cls, zone):
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

    @classmethod
    def __preprocess_rawdata(cls, raw_data, probe_item_list, probe_item, probe_dependent_item_list,
                             esda_items, esda_items_1, esda_items_d, 
                             extra_esda_items):
        pareto_pct = cls.TOP_ESDA_PARETO_PCT
        """
        This function preprocess the raw data by filtering out the failed dies and convert the data to numerical type.
        :param raw_data: slightly processed raw data in pandas Dataframe
        :param probe_item_list: probe column in the dataframe, e.g. ['__rdC'], ['__rdV']
        :param probe_item: probe column from input, e.g. 'V', 'C' etc
        :param probe_dependent_item_list: probe column in the dataframe, e.g. ['__rdt'], ['__rdZ']
        :param esda_items: ESDA columns in the dataframe, e.g. ['BITR0', 'BITR1', ...]
        :param esda_items_1: ESDA columns tied to probe_item, e.g. ['BITR0_V_ALL', 'BITR1_V_ALL', ...]
        :param esda_items_d: ESDA columns tied to probe_dependent_item, e.g. ['BITR0_t_ALL', 'BITR1_t_ALL', ...]
        :param extra_esda_items: extra ESDA columns ['DBAWPA_VS_ALL, DBTT_V2_ALL']
        return: processed pandas dataframe
        """
        if not isinstance(raw_data, pd.DataFrame):
            raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "raw_data is not a pandas dataframe.")
            
        if len(raw_data)==0:
            print("raw_data is empty")
            raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "queried data is empty.")

        esda_items_all = esda_items_1 + esda_items_d + extra_esda_items
        probe_items_all = probe_item_list + probe_dependent_item_list
        start_time = time.time()

        esda_items_selected, esda_items_mix = [], []
        #Clean up empty esda columns and convert to float
        empty_cols = raw_data[esda_items_all].columns[raw_data[esda_items_all].nunique() <= 1].tolist()
        print("empty or nonvariant columns: ", empty_cols)
        if len(empty_cols) == len(esda_items_all):
            raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "all ESDA columns are empty.")
        raw_data = raw_data[list(set(raw_data.columns) - set(empty_cols))]
        del_row_index = []
        for item in set(esda_items_all+probe_items_all)-set(empty_cols):
            df_1 = raw_data[item]
            fail_die_index = df_1.index[df_1.isin(['X'])].tolist()
            del_row_index = del_row_index+fail_die_index
        del_row_index_unique = np.unique(del_row_index)
        index_selected = list(set(raw_data.index) - set(del_row_index_unique))
        raw_data = raw_data.iloc[index_selected]
        print('%d failed dies deleted' %(len(del_row_index_unique)))
        print('%d dies left' %(raw_data.shape[0]))

        esda_items_1 = list(set(esda_items_1)-set(empty_cols))
        esda_items_1.sort()
        raw_1 = raw_data[esda_items_1]
        raw_1 = raw_1.astype(float)
        raw_rd = raw_data[list(set(raw_data.columns)-set(esda_items_all))]
        raw_rd['FBD_REGION'] = raw_rd['FBD_REGION'].apply(lambda x: cls.__label_zone(x))
        if 'BIN_BIT' in probe_items_all:
            raw_rd['BIN_BIT'] = np.where(raw_rd['BIN_BIT'].str.endswith(':'+probe_item), 1, 0)
            raw_rd = raw_rd[raw_rd['BIN_BIT']==1] # Only keep data which fail the target bin
        raw_rd[probe_items_all] = raw_rd[probe_items_all].astype(float)

        esda_items_selected = esda_items_1

        #If rd dependent item is not null, prep the ESDA columns with respect to rd and dependent rd
        if len(esda_items_d)>0:
            esda_items_d = list(set(esda_items_d)-set(empty_cols))
            esda_items_d.sort()
            raw_d = raw_data[esda_items_d]
            raw_d = raw_d.astype(float)
            rd_1 = probe_item_list[0][-1]
            rd_d = probe_dependent_item_list[0][-1]
            rd_1_1d = rd_1 + '-' + rd_d
            for esda_item_1, esda_item_d in zip(esda_items_1, esda_items_d):
                if esda_item_1.startswith(esda_item_d.split('_')[0]):
                    esda_item_mix = esda_item_1.replace('_'+rd_1+'_', '_'+rd_1_1d+'_')
                    raw_rd[esda_item_mix] = np.where((raw_1[esda_item_1] - raw_d[esda_item_d]) < 0.0, 
                                                        0.0, 
                                                    (raw_1[esda_item_1] - raw_d[esda_item_d]))
                    esda_items_mix.append(esda_item_mix)
                else:
                    print('ECAT %s not matched with Dependent ECAT %s.' %(esda_item_1, esda_item_d))
            esda_items_selected = esda_items_mix
        else:
            raw_rd = pd.concat([raw_rd, raw_1], axis=1)

        if len(extra_esda_items)>0:
            esda_items_e = list(set(extra_esda_items)-set(empty_cols))
            raw_e = raw_data[esda_items_e]
            raw_e = raw_e.astype(float)
            esda_items_selected.extend(esda_items_e)
            raw_rd = pd.concat([raw_rd, raw_e], axis=1)
        
        print(raw_rd.head())
        #convert zone column to integer
        cluster_col_sort = raw_rd[esda_items_selected].sum().sort_values(ascending=False)
        cluster_col_contr = cluster_col_sort/cluster_col_sort.sum()
        cluster_col_contr_cum = cluster_col_contr.cumsum()
        cluster_col_pareto = cluster_col_contr_cum[cluster_col_contr_cum<pareto_pct].index.tolist()
        percent_pareto = pareto_pct
        #handle inputs with too few ECAT columns
        if len(cluster_col_pareto)<6:
            limit_len = min(8, len(cluster_col_contr_cum.index.tolist()))
            cluster_col_pareto = cluster_col_contr_cum.index.tolist()[0:limit_len]
            print('cluster cols: ', cluster_col_pareto)
            percent_pareto = cluster_col_contr_cum.loc[cluster_col_pareto[-1]]
        print("top %.1f%% ECATs: %s" %(percent_pareto*100, cluster_col_pareto))
        #check if probe_item is a probe bin, if yes use it as additional item for clustering
        if len(probe_item_list[0]) ==1:
            cluster_col_pareto.extend(probe_item_list)
        print('Pre-Processing time = ', time.time() - start_time)
        return raw_rd, cluster_col_pareto, esda_items_selected
