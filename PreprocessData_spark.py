import pandas as pd
import numpy as np
import time
from pyspark.sql.functions import when, col, split, udf, lit
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
import sys

class ProbeDataPreprocess(object):
    TOP_ESDA_PARETO_PCT = 0.99
    MIN_VAR_COUNT = 8
    @classmethod
    def dataprep_pyspark(cls, probeDF, rd_item, rd_dependent_item,
                           esda_items, esda_items_1, esda_items_d, 
                           extra_esda_items, features, bin_step):
        """
        This function preprocess the raw data by filtering out the failed dies and convert the data to numerical type.
        :param probeDF: raw data in spark Dataframe
        :param rd_item: rd column in the dataframe, e.g. ['__rdC'], ['__rdV']
        :param rd_dependent_item: rd column in the dataframe, e.g. ['__rdt'], ['__rdZ']
        :param esda_items: ESDA columns in the dataframe, e.g. ['BITR0', 'BITR1', ...]
        :param esda_items_1: ESDA columns tied to rd_item, e.g. ['BITR0_V_ALL', 'BITR1_V_ALL', ...]
        :param esda_items_d: ESDA columns tied to rd_dependent_item, e.g. ['BITR0_t_ALL', 'BITR1_t_ALL', ...]
        :param extra_esda_items: extra ESDA columns ['DBAWPA_VS_ALL, DBTT_V2_ALL']
        """
        '''split FID column into lot,wafer,dieX and dieY in a small df
            create lot_wafer column from lot and wafer
            merge back into main df'''
        start_time = time.time()
        if probeDF.rdd.isEmpty():
            print("raw_data is empty")
            raise Exception("PLUNK_LOGGING_MAPA:Failed|" + "queried data is empty.")
            sys.exit()
        else:
            esda_items_all = esda_items_1 + esda_items_d + extra_esda_items
            rd_items_all = rd_item + rd_dependent_item
            all_registers = esda_items_all + rd_items_all + features
            all_registers = list(set(all_registers))
            if 'BIN_BIT' in all_registers:
                probeDF = probeDF.filter(~probeDF.BIN_BIT.startswith('.:'))
                all_registers.remove('BIN_BIT')
            elif bin_step is not None:
                probeDF = probeDF.filter(~F.col(bin_step).contains('NA!'))
                probeDF = probeDF.filter(~F.col(bin_step).contains('.'))
                all_registers.remove(bin_step)
            probeDF = probeDF.filter(probeDF.REPROBE == '0')
            probe_df = probeDF.select(split(probeDF.FID,":")).rdd.flatMap(
                                                      lambda x: x).toDF(schema=['LotId', 'WaferId', 'DieX', 'DieY'])
            probe_df = probe_df.withColumn('DieX', F.when(
                                            F.col('DieX').startswith('N'),
                                            F.concat(lit('-'), F.col('DieX').substr(2, 4))
                                            ).otherwise(F.col('DieX').substr(2, 4)))
            probe_df = probe_df.withColumn("DieX", probe_df["DieX"].cast("Integer"))
            probe_df = probe_df.withColumn("DieY", probe_df["DieY"].cast("Integer"))
            probe_df = probe_df.withColumn("lot_wafer", F.concat(col('LotId'), lit("_"), col('WaferId')))
            '''create a row_id for merging and then discard'''
            probeDF = probeDF.withColumn("row_id", F.monotonically_increasing_id())
            probe_df = probe_df.withColumn("row_id", F.monotonically_increasing_id())
            probeDF = probeDF.join(probe_df, ("row_id")).drop("row_id")
            '''apply countDistinct on each column'''
            col_counts = probeDF.agg(*(F.countDistinct(col(c)).alias(c) for c in all_registers)).collect()[0].asDict()
            '''select the cols with count=1 in an array'''
            cols_to_drop = [col for col in all_registers if 
                            (col_counts[col] == 1) or (col_counts[col] == 2)]
            cols_to_drop = [e for e in cols_to_drop if e not in features]
            '''drop the selected non-variating columns'''
            # will never drop probe_item even if it's either 0 or "X"
            cols_to_drop = list(set(cols_to_drop) - set(rd_items_all))
            print('columns dropped: ', cols_to_drop)
            col_esda_remain = list(set(esda_items_all) - set(cols_to_drop))
            print("esda columns remained: ", col_esda_remain)
            if len(col_esda_remain) == 0:
                print("esda columns are empty, no data left for clustering.")
                raise Exception("PLUNK_LOGGING_MAPA:Failed|" + "esda data is empty, no data left for clustering.")
                sys.exit()
            probeDF = probeDF.drop(*cols_to_drop)
            '''call udf for wafer zone encoding'''
            #zone_udf = udf(cls.__label_zone, StringType())
            #probeDF = probeDF.withColumn("FBD_REGION", zone_udf(probeDF['FBD_REGION']))
            probeDF = probeDF.withColumn("FBD_REGION", 
                                         when(col("FBD_REGION")=='A', 
                                              1).when(col("FBD_REGION")=='B', 
                                                2).when(col("FBD_REGION")=='C', 
                                                3).when(col("FBD_REGION")=='D', 
                                                4).when(col("FBD_REGION")=='E', 
                                                5).otherwise(0))
            '''get register columns with variations'''
            filter_cols = list(set(all_registers) -  set(cols_to_drop))
            '''get df with any empty register in a row'''
            probe_df = probeDF.where("OR".join(["(%s =='X')"%(col) for col in filter_cols]))
            '''get df with none empty register in a row'''
            probeDF = probeDF.where("AND".join(["(%s !='X')"%(col) for col in filter_cols]))
            print('%d failed dies deleted' %(probe_df.count()))
            '''convert registers to float for passed dies'''
            for col_name in filter_cols:
                probeDF = probeDF.withColumn(col_name, col(col_name).cast('float'))
            df, cluster_col_pareto, esda_items_selected = cls.__preprocess_rawdata(probeDF, 
                rd_item, rd_dependent_item, esda_items, esda_items_1, esda_items_d, extra_esda_items, cols_to_drop)

        print('Pre-Processing time = ', time.time() - start_time)
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
    def __preprocess_rawdata(cls, probeDF, rd_item, rd_dependent_item,
                             esda_items, esda_items_1, esda_items_d, 
                             extra_esda_items, cols_to_drop):
        pareto_pct = cls.TOP_ESDA_PARETO_PCT
        min_var_count = cls.MIN_VAR_COUNT
        """
        This function preprocess the raw data by filtering out the failed dies and convert the data to numerical type.
        :param probeDF: slightly processed raw data in spark Dataframe
        :param rd_item: rd column in the dataframe, e.g. ['__rdC'], ['__rdV']
        :param rd_dependent_item: rd column in the dataframe, e.g. ['__rdt'], ['__rdZ']
        :param esda_items: ESDA columns in the dataframe, e.g. ['BITR0', 'BITR1', ...]
        :param esda_items_1: ESDA columns tied to rd_item, e.g. ['BITR0_V_ALL', 'BITR1_V_ALL', ...]
        :param esda_items_d: ESDA columns tied to rd_dependent_item, e.g. ['BITR0_t_ALL', 'BITR1_t_ALL', ...]
        :param extra_esda_items: extra ESDA columns ['DBAWPA_VS_ALL, DBTT_V2_ALL']
        :param cols_to_drop: columns to drop due to no varition or empty data
        return: processed pandas dataframe
        """
        esda_items_selected, esda_items_mix = [], []
        esda_items_1 = list(set(esda_items_1)-set(cols_to_drop))
        esda_items_selected = esda_items_1

        #If rd dependent item is not null, prep the ESDA columns with respect to rd and dependent rd
        if len(esda_items_d)>0:
            esda_items_mix = []
            esda_items_d = list(set(esda_items_d)-set(cols_to_drop))
            esda_items_d.sort()
            rd_1 = rd_item[0][-1]
            rd_d = rd_dependent_item[0][-1]
            rd_1_1d = rd_1 + '-' + rd_d
            for esda_item in esda_items:
                esda_item_1 = [e for e in esda_items_1 if e.startswith(esda_item)]
                esda_item_d = [e for e in esda_items_d if e.startswith(esda_item)]
                if len(esda_item_1)>0 and len(esda_item_d)>0:
                    esda_item_1, esda_item_d = esda_item_1[0], esda_item_d[0]
                    esda_item_mix = esda_item_1.replace('_'+rd_1+'_', '_'+rd_1_1d+'_')
                    probeDF = probeDF.withColumn(esda_item_mix, col(esda_item_1) - col(esda_item_d))
                    probeDF = probeDF.withColumn(esda_item_mix, F.when(col(esda_item_mix)<0.0, lit(0.0)
                                                                        ).otherwise(col(esda_item_mix)))
                    probeDF = probeDF.drop(*[esda_item_1, esda_item_d])
                    esda_items_mix.append(esda_item_mix)
            esda_items_selected = esda_items_mix
        if len(extra_esda_items)>0:
            esda_items_e = list(set(extra_esda_items)-set(cols_to_drop))
            esda_items_selected.extend(esda_items_e)
        
        #get top pareto for cluster columns
        cluster_col_sum = probeDF.agg(*(F.sum(col(c)).alias(c) for c in esda_items_selected)).collect()[0].asDict()
        dict_sort = sorted(cluster_col_sum.items(), key=lambda kv: kv[1], reverse=True)
        col_pareto = [col[0] for col in dict_sort]
        val_pareto = [col[1] for col in dict_sort]
        val_cumsum = [np.sum(val_pareto[0:i+1]) for i in np.arange(len(val_pareto))]/np.sum(val_pareto)
        val_cumsum_1 = val_cumsum[val_cumsum<pareto_pct]
        #select min num variables based on data & cutoff threshold
        min_len = min(len(val_cumsum_1)+1, len(val_pareto))
        if min_len < 6:
            min_len = min(min_var_count, len(col_pareto))
        col_pareto = col_pareto[0:min_len]
        pnt_pareto = val_cumsum[min_len-1]
        print("top %.1f%% ECATs: %s" %(pnt_pareto*100, col_pareto))

        return probeDF, col_pareto, esda_items_selected
