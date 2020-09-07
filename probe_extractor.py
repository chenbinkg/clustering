import sys
import os
import numpy as np
import pandas as pd
import logging
import json
import datetime
import math
import time
import re

class ProbeExtractor(object):

    DEFAULT_PROBE_STEP = "FPP"
    RD_PREFIX = "__"
    ESDA_SUFFIX = "_ALL"
    ESDA_PREFIX = "_"
    DID_GROUP = {
		"DRAM": "U,V,X,Y,Z", 
		"NAND": "B,F,G,L,M,N",
        "3DXP": "S,H"
	    }
    
    @classmethod
    def get_esda_setting(cls, did, rd_item, esda_items, bin_step=None, rd_dependent_item=None, extra_esda_items=None):
        """
        did: like 'Z22A'
        rd_item: like 'rdC', 'rdV'
        esda_items: raw esda list like ['BDGWD','BITCR0','BITCR1', ...]
        rd_dependent_item (optional): same format as rd_item such as 'rdt', 'rdZ', 'rdV'
        extra_esda_items (optional): comma separated full esda_items like 'DBAWPA_VS_ALL,DBTT_V2_ALL'
        
        Return: 
        rd_item like ['__rdC']
        esda_items list like ['BDGWD_V_ALL','BITCR0_V_ALL','BITCR1_V_ALL', ...] 
        rd_dependent_item like ['__rdt'] or []
        dependent_esda_items list like ['BDGWD_t_ALL','BITCR0_t_ALL','BITCR1_t_ALL', ...] or []
        extra_esda_items: ['DBAWPA_VS_ALL, DBTT_V2_ALL'] or []
        """
        did_group = cls.DID_GROUP
        dram_did = did_group['DRAM'].split(',')
        dram_did = [e.strip() for e in dram_did]
        nand3d_did = did_group['NAND'].split(',')
        nand3d_did = [e.strip() for e in nand3d_did]
        nvm3dxp_did = did_group['3DXP'].split(',')
        nvm3dxp_did = [e.strip() for e in nvm3dxp_did]
        if did[0] in dram_did:
            did_key = 'DRAM'
        elif did[0] in nand3d_did:
            did_key = 'NAND'
        elif did[0] in nvm3dxp_did:
            did_key = '3DXP'
        else:
            did_key = 'OTHER'
        #Prep rd_item format and esda_items into full form with rd bin
        rd_prefix, esda_suffix, esda_prefix = cls.RD_PREFIX, cls.ESDA_SUFFIX, cls.ESDA_PREFIX
        esda_base_reg = [e.strip() for e in esda_items]
        #Check rd_item format
        rd_maching_list = ['rd']
        if rd_item.startswith(tuple(rd_maching_list)):
            match_str = [e for e in rd_maching_list if rd_item.startswith(e)][0]
            rd_bin = rd_item[len(match_str):]
            ecat_suffix = esda_prefix + rd_bin + esda_suffix
            esda_items = esda_base_reg
            rd_item = [rd_prefix+rd_item]
        elif bin_step == "Fail_bin":
            rd_bin = rd_item
            ecat_suffix = esda_prefix + rd_bin
            esda_items = esda_base_reg
            rd_item = ['BIN_BIT']
        elif (bin_step != "Fail_bin") and (bin_step is not None):
            rd_bin = rd_item
            ecat_suffix = esda_prefix + rd_bin
            esda_items = esda_base_reg
            rd_item = [bin_step]
        else:
            raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "probe item not properly defined")
            sys.exit()
        #Prep rd_dependent_item format and dependent_esda_items using esda_items as the base
        if rd_dependent_item is not None:
            if rd_dependent_item.startswith(tuple(rd_maching_list)):
                match_str = [e for e in rd_maching_list if rd_dependent_item.startswith(e)][0]
                rd_dependent_bin = rd_dependent_item[len(match_str):]
                rd_dependent_bin_suffix = esda_prefix + rd_dependent_bin + esda_suffix
                dependent_esda_items = [e.split(ecat_suffix)[0] + rd_dependent_bin_suffix for e in esda_items]
                rd_dependent_item = [rd_prefix+rd_dependent_item]
            else:
                raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "dependent rd item not properly defined")
                sys.exit()
        else:
            rd_dependent_item, dependent_esda_items = [], []
        #Change extra_esda_items into list, assuming it's keyed in using full form
        if extra_esda_items is not None:
            extra_esda_items = extra_esda_items.split(',')
            extra_esda_items = [e.strip() for e in extra_esda_items]
        else:
            extra_esda_items = []
        #remove duplicate ECATs case sensitively
        esda_items, dependent_esda_items, extra_esda_items = cls.__remove_dupes_ecat(esda_items, 
                                                                                     dependent_esda_items, 
                                                                                     extra_esda_items)
        
        return rd_item, esda_items, rd_dependent_item, dependent_esda_items, extra_esda_items, did_key
    
    @classmethod
    def extract_probe_data_by_lot(
        cls,
        lot_list,
        principal,
        keytab,
        site,
        esda_columns,
        rd_columns,
        logger=None,
        yarn_queue="eng_oct",
        pid="FPP"
    ):
        """
        : param lot_list: lot id list 
        : param principal: email for the keytab
        : param keytab: file path of the keytab
        : param site: The same as mu_probeextract like 'fab15'
        : param esda_columns: e.g. ['BITR0_V_ALL', 'BITR1_V_ALL', ...]
        : param rd_columns: e.g. ['__rdV', '__rdC'] etc
        : param logger: logger for logging
        : return : A Pandas Dataframe which includes columns
        """
        start_time = time.time()
        if logger is None:
            # get a default logger
            logger = logging.getLogger()
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)

        logger.info(str(datetime.datetime.now()))

        lot_list = [x[:7] for x in lot_list]
        lot_list = list(set(lot_list))

        executor_number = math.ceil(len(lot_list) / 2)
        executor_number = 256 if executor_number > 256 else executor_number

        esda_columns = [x.strip() for x in esda_columns]

        logger.info("esda_columns: " + ",".join(esda_columns))
        logger.info("rd_columns: " + ",".join(rd_columns))

        sc, pc = cls.get_mu_probe_context(principal, keytab, executor_number, yarn_queue, logger)

        query_register_list = esda_columns + rd_columns
        registers = ",".join(query_register_list)
        registers = "FID,DESIGN_ID,PART_TYPE,FAB,RUN_ID,LOT_ID,WAFER_ID,FBD_REGION,REPROBE,START_DATETIME,TOTAL_DIE," + registers
        print("querying rd: " + ",".join(rd_columns))
        print("registers: ", registers)
        probe_df = pc.get_probe_data_by_lot_process(
            site=site, lots=lot_list, fields=registers, process=pid, orientation="wide"
        )
        print('query completed, time taken: %.1f secs' %(time.time()-start_time))
        return probe_df, sc

    @classmethod
    def extract_probe_data_by_datetime(
        cls,
        start_date,
        end_date,
        site,
        did,
        principal,
        keytab,
        esda_columns,
        rd_columns,
        logger=None,
        yarn_queue="eng_oct",
        pid="FPP"
    ):
        """
        : param start_date: start date like '2019-06-04' or '190604'
        : param end_date: end date like '2019-06-04' or '190604'
        : param site: The same as mu_probeextract like 'fab15'
        : param did: The same as mu_probeextract like 'Z22A'
        : param principal: email for the keytab
        : param keytab: file path of the keytab
        : param esda_columns: e.g. ['BITR0_V_ALL', 'BITR1_V_ALL', ...]
        : param rd_columns: e.g. ['__rdV', '__rdC'] etc
        : param logger: logger for logging
        : return : A Pandas Dataframe which includes columns
        """
        start_time = time.time()
        if logger is None:
            # get a default logger
            logger = logging.getLogger()
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)

        start_date = cls.__make_datetime_format(start_date)
        end_date = cls.__make_datetime_format(end_date)

        logger.info(str(datetime.datetime.now()))

        executor_number = 256

        esda_columns = [x.strip() for x in esda_columns]
        esda_string = ",".join(esda_columns)
        rd_string = ",".join(rd_columns)
        logger.info("esda_columns: " + esda_string)
        logger.info("rd_columns: " + rd_string)

        sc, pc = cls.get_mu_probe_context(principal, keytab, executor_number, yarn_queue, logger)

        query_register_list = esda_columns + rd_columns
        registers = ",".join(query_register_list)
        registers = "FID,DESIGN_ID,PART_TYPE,FAB,RUN_ID,LOT_ID,WAFER_ID,FBD_REGION,REPROBE,START_DATETIME,TOTAL_DIE," + registers
        # probe_df = pc.get_probe_data_by_lot(lot_list, registers , 'tall')
        print("querying rd: " + ",".join(rd_columns))
        print("registers: ", registers)
        probe_df = pc.get_probe_data_by_timerange_process(
            did=did, site=site, start_time=start_date, end_time=end_date, registers=registers, orientation="wide", process=pid
        )
        print('query completed, time taken: %.1f secs' %(time.time()-start_time))
        return probe_df, sc

    @classmethod
    def get_mu_probe_context(cls, principal, keytab, executor_number, yarn_queue, logger):

        import mu_pysparksetup

        ss = mu_pysparksetup.MuPySparkSetup("/usr/hdp/current", spark_version=2)

        logger.info("mu_pysparksetup version is:" + str(mu_pysparksetup.__version__))
        logger.info("SPARK VERSION: " + str(ss.spark_version))

        import mu_hbase4pyspark
        import mu_probeextract
        import mu_productlayout

        logger.info(
            """
mu_hbase4pyspark version is {}
mu_productlayout version is {}
mu_probeextract version is {}
""".format(
                mu_hbase4pyspark.__version__, mu_productlayout.__version__, mu_probeextract.__version__
            )
        )

        DEP_JARS = [mu_hbase4pyspark.get_jars(), mu_probeextract.get_jars(), mu_productlayout.get_jars()]

        principal = "hdfsoct@NA.MICRON.COM" if principal is None else principal
        keytab = "/home/hdfsoct/.keytab/hdfsoct.keytab" if keytab is None else keytab

        config = {
            "spark.dynamicAllocation.enabled": "true",
            "spark.shuffle.service.enabled": "true",
            "spark.dynamicAllocation.executorIdleTimeout": "5m",
#             "spark.dynamicAllocation.minExecutors": "200",
            "spark.dynamicAllocation.initialExecutors": "200",
            "spark.dynamicAllocation.maxExecutors": "400",
#             'spark.sql.execution.arrow.enabled': 'true',
            # 'spark.sql.execution.arrow.maxRecordsPerBatch': '1000',
            "spark.executor.memory": "70g",
            "spark.executor.memoryOverhead": "8g",
            "spark.executor.cores": "5",
            "spark.driver.memory": "32g",
            "spark.driver.maxResultSize": "40g",
            "spark.yarn.queue": yarn_queue,
            "spark.default.parallelism": "200",
            "spark.sql.shuffle.partitions": "400",
            "spark.yarn.principal": principal,
            "spark.yarn.keytab": keytab,
            "spark.app.name": "ESDAKEMClustering_DataExtractor",
            "spark.master": "yarn",
        }

        sc = ss.init(
            name="ESDAKEMClustering_DataExtractor",
            master="yarn-client",
            config_parameters=config,
            dependent_jars=DEP_JARS,
        )

        sc.sparkContext.setLogLevel("ERROR")
        hbc = mu_hbase4pyspark.HBaseContext(sc.sparkContext, principal, keytab)
        pc = mu_probeextract.ProbeContext(hbc)
        return sc, pc

    @classmethod
    def __make_datetime_format(cls, dt_str):
        """
        Change the original date time string to the format accepted by mu_probeextract. which is like '190604'
        Keyword arguments:
        dt_str -- datetime string. expected to be '2019-06-04'
        Return: datetime string like '190604'
        """

        dt_str = dt_str[:10]
        if re.match("\d{4}-\d{2}-\d{2}", dt_str) is not None:
            return dt_str[2:4] + dt_str[5:7] + dt_str[8:10]
        elif re.match("\d{6}", dt_str[:6]):
            return dt_str[:6]
        else:
            raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "Incorrect datetime format")
            sys.exit()
    
    @classmethod
    def __remove_dupes_ecat(cls, esda_items_1, esda_items_d, extra_esda_items):
        esda_items_1_l = [e.lower() for e in esda_items_1]
        esda_items_d_l = [e.lower() for e in esda_items_d]
        esda_items_1_f, dlist_1 = cls.__remove_dupes(esda_items_1_l, esda_items_1)
        esda_items_d_f, dlist_d = cls.__remove_dupes(esda_items_d_l, esda_items_d)
        extra_esda_items_f = list(set(extra_esda_items) - set(esda_items_1+esda_items_d))
        if len(dlist_1)>0:
            print('Duplicate main ecats removed: ', dlist_1)
        if len(dlist_d)>0:
            print('Duplicate dependent ecats removed: ', dlist_d)
        if (len(extra_esda_items) - len(extra_esda_items_f))>0:
            print('Duplicate extra ecats removed: ', list(set(extra_esda_items) - set(extra_esda_items_f)))
        return esda_items_1_f, esda_items_d_f, extra_esda_items_f
    
    @classmethod
    def __remove_dupes(cls, l, o):
        seen = {}
        dupes = []
        flist = []
        dlist = []

        for (x, y) in zip(l, o):
            if x not in seen:
                seen[x] = 1
                flist.append(y)
            else:
                if seen[x] == 1:
                    dupes.append(x)
                dlist.append(y)
                seen[x] += 1
        return flist, dlist
