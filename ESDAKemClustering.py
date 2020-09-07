import os
import pandas as pd
import numpy as np
import sys
from mu_esda_kemclustering.probe_extractor import ProbeExtractor
from mu_esda_kemclustering.kem_clustering import KEMClustering
from mu_esda_kemclustering.PreprocessData_spark import ProbeDataPreprocess
from mu_esda_kemclustering.StackedWaferMap import CalStackedWaferMap
from mu_esda_kemclustering.ESDAKemLogging import ESDAKemLogging
from mu_esda_kemclustering.label_failure_modes import LabelData
from mu_esda_kemclustering.plot_ecat_corr import PlotECATCorr
import logging
'''Turn off warning'''
pd.options.mode.chained_assignment = None

class ESDAKemClustering(object):

    DELIMITER = "::"

    def _setup_output_mode(self, output_mode, csv_output_dir=None):
        """
        : param output_mode: 'csv', 'dataframe', 'df_4_debug' or a list containing multiple choice
        : param csv_output_dir: If 'csv' output is selected, this path must be provided.
        : param logging_file_name: file name of the logging output file
        """
        self.output_dataframe = False
        self.output_csv = False
        self.output_debug_df = False
        self.stacked_map_file_name = "cluster_signature.json"
        self.wafer_esda_pareto_file_name = "wafer_esda_pareto.json"
        self.cluster_esda_pareto_file_name = "cluster_esda_pareto.json"
        self.clustering_result_file_name = "cluster_wafer_association.csv"
        self.clustering_result_json_file_name = "cluster_wafer_association.json"
        if isinstance(output_mode, list):
            if "csv" in output_mode:
                if csv_output_dir is not None:
                    self.output_csv = True
                    self.csv_output_dir = csv_output_dir
                else:
                    raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "CSV output selected but not given csv output path")
                    sys.exit()
            if "dataframe" in output_mode:
                self.output_dataframe = True
            if "df_4_debug" in output_mode:
                self.output_debug_df = True
        elif output_mode == "csv":
            if csv_output_dir is not None:
                self.output_csv = True
                self.csv_output_dir = csv_output_dir
            else:
                raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "CSV output selected but not given csv output path")
                sys.exit()
        elif output_mode == "dataframe":
            self.output_dataframe = True
        elif output_mode == "df_4_debug":
            self.output_debug_df = True

    def run(
        self,
        job_id,
        did,
        probe_item,
        csv_output_dir,
        site,
        esda_items,
        bin_step=None,
        probe_dependent_item=None,
        extra_esda_items=None,
        lot_list=None,
        start_time=None,
        end_time=None,
        model_dir="file:///home/hdfsf15/pg1_esda/mu_esda_kemclustering/model/",
        plot_result_dir = "/home/hdfsf10w/pg1_esda/mu_esda_kemclustering/result/",
        pid="FPP",
        output_mode="csv",
        user_principal="hdfsoct@NA.MICRON.COM",
        user_keytab="/home/hdfsoct/.keytab/hdfsoct.keytab",
        queue="eng_oct"
    ):
        """
        : param job_id: job_id from piper
        : param did: design id e.g. 'Z32M'
        : param probe_item: single rd bin column for stacked wafer map, e.g. 'rdV', 'rdC' or 'V', 'C' etc
        : param site: fab, e.g. 'fab15' or 'fab10'
        : param esda_items: esda items to cluster, compulsory input like ['BDGWD','BITCR0','BITCR1', ...], read from UI
        : param probe_dependent_item: single rd bin column for stacked wafer map, e.g. 'rdV', 'rdC', or 'V', 'C'
        : param extra_esda_items: extra esda items in comman separated string format, e.g. 'DBAWPA_VS_ALL,DBTT_V2_ALL'
        : param pid: e.g. 'FPP', 'FQQP', default is 'FPP'
        : param lot_list: list of lots
        : param start_time: 'YYYY-MM-DD HH:MM:SS', '2019-06-04'
        : param end_time: '2019-06-05'
        : param principal: account
        : param keytab: kerboros keytab
        : return: Pandas Dataframe which contains original data and columns with 
        """
        try:
            self._setup_output_mode(output_mode, csv_output_dir)

            log_path = os.path.join(self.csv_output_dir, "log")
            ESDAKemLogging(job_id, log_path)

            return self._run(
                job_id=job_id,
                esda_items=esda_items,
                probe_dependent_item=probe_dependent_item,
                extra_esda_items=extra_esda_items,
                probe_item=probe_item,
                bin_step=bin_step,
                principal=user_principal,
                keytab=user_keytab,
                lot_list=lot_list,
                site=site.lower().replace(" ", ""),
                did=did,
                start_time=start_time,
                end_time=end_time,
                queue=queue,
                pid=pid,
                model_dir=model_dir,
                plot_result_dir=plot_result_dir
            )
        except:
            ESDAKemLogging.getInstance().exception("SPLUNK_LOGGING_MAPA:Failed|" + "Exception in main function")
            raise
            sys.exit()

    def _run(
        self,
        job_id,
        probe_item,
        principal,
        keytab,
        queue,
        esda_items,
        site,
        bin_step=None,
        model_dir=None,
        plot_result_dir=None,
        pid=None,
        probe_dependent_item=None,
        extra_esda_items=None,
        lot_list=None,
        did=None,
        start_time=None,
        end_time=None
    ):
        """
        : param job_id: job_id from piper
        : param did: design id e.g. 'Z32M'
        : param probe_item: single rd bin column for stacked wafer map, e.g. 'rdV', 'rdC', or 'V', 'C'
        : param site: site, e.g. 'fab15', or 'fab10'
        : param esda_items: esda items to cluster, compulsory input like ['BDGWD','BITCR0','BITCR1', ...], read from UI
        : param probe_dependent_item: single rd bin column for stacked wafer map, e.g. 'rdV', 'rdC', or 'V', 'C'
        : param extra_esda_items: extra esda items in comman separated string format, e.g. 'DBAWPA_VS_ALL,DBTT_V2_ALL'
        : param pid: e.g. 'FPP', 'FQQP', default is 'FPP'
        : param lot_list: list of lots
        : param start_time: 'YYYY-MM-DD HH:MM:SS', '2019-06-04'
        : param end_time: '2019-06-05'
        : param principal: account
        : param keytab: kerboros keytab
        : return: Pandas Dataframe which contains original data and columns with 
        """

        raw_df = None
        probe_item_list, esda_items_1, probe_dependent_item_list, \
            esda_items_d, extra_esda_items, did_key = ProbeExtractor.get_esda_setting(did, probe_item, esda_items, bin_step,
                                                                                      probe_dependent_item, extra_esda_items)
        features, extra_ecats = LabelData.read_model_ecat(site, did, probe_item, esda_items_1, esda_items_d, 
                                                          extra_esda_items, model_dir)
        esda_items_all = esda_items_1 + esda_items_d + extra_esda_items + extra_ecats
        probe_items_all = probe_item_list + probe_dependent_item_list
        
        if isinstance(lot_list, list):
            if site is None:
                raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "No site parameter provided when querying with lot_list")
                sys.exit()
            try:
                raw_df, sc = ProbeExtractor.extract_probe_data_by_lot(
                    lot_list, 
                    principal, 
                    keytab, 
                    site, 
                    esda_items_all, 
                    probe_items_all, 
                    logger=ESDAKemLogging.getInstance(), 
                    yarn_queue=queue, 
                    pid=pid
                )
            except Exception as e:
                print("SPLUNK_LOGGING_MAPA:Failed|" + "At query_by_lotlist stage; " + str(sys.exc_info()[1]) + "|")
                sys.exit()
        elif None not in (site, did, start_time, end_time):
            try:
                raw_df, sc = ProbeExtractor.extract_probe_data_by_datetime(
                    start_time,
                    end_time,
                    site,
                    did,
                    principal,
                    keytab,
                    esda_items_all,
                    probe_items_all,
                    logger=ESDAKemLogging.getInstance(),
                    yarn_queue=queue,
                    pid=pid
                )
            except Exception as e:
                print("SPLUNK_LOGGING_MAPA:Failed|" + "At query_by_datetime stage; " + str(sys.exc_info()[1]) + "|")
                sys.exit()
        else:
            raise Exception("SPLUNK_LOGGING_MAPA:Failed|" + "incorrect input parameters for _run")
            sys.exit()

        # Clustering and save stacked wafermap
        try:
            preprocessed_df, cluster_cols, esda_items_cols = ProbeDataPreprocess.dataprep_pyspark(raw_df, probe_item_list, 
                                                                                                  probe_dependent_item_list,
                                                                                                  esda_items, esda_items_1, 
                                                                                                  esda_items_d, extra_esda_items, 
                                                                                                  features, bin_step)
        except Exception as e:
            sc.stop()
            print("SPLUNK_LOGGING_MAPA:Failed|" + "At dataprep stage; " + str(sys.exc_info()[1]) + "|")
            sys.exit()
        
        try: 
            df_labelled, df_unlabelled, labelled_failure_modes = LabelData.label_failure_modes(site, did, 
                                                                                               probe_item, 
                                                                                               preprocessed_df, 
                                                                                               model_dir, sc)
        except Exception as e:
            sc.stop()
            print("SPLUNK_LOGGING_MAPA:Failed|" + "At failure mode labelling stage: " + str(sys.exc_info()[1]) + "|")
            sys.exit()
#        sc, pc = ProbeExtractor.get_mu_probe_context(principal, keytab, 64, yarn_queue=queue, logger=ESDAKemLogging.getInstance())
        
        try:
            df_unlabelled, clustering_models, stacked_items_list = KEMClustering.run_mv_clustering(df_unlabelled, cluster_cols, 
                                                                                                    did_key, probe_item_list, sc, 
                                                                                                    probe_item, bin_step)
        except Exception as e:
            sc.stop()
            print("SPLUNK_LOGGING_MAPA:Failed|" + "At clustering stage: " + str(sys.exc_info()[1]) + "|")
            sys.exit()
        
        try:
            stacked_map_df, cluster_pareto_df = CalStackedWaferMap.generate_stacked_wafer_map(df_labelled, labelled_failure_modes,
                                                                                                df_unlabelled, clustering_models, 
                                                                                                probe_item_list, probe_item, bin_step,
                                                                                                cluster_cols, stacked_items_list)
        except Exception as e:
            print("SPLUNK_LOGGING_MAPA:Failed|" + "At stacked wafermap stage: " + str(sys.exc_info()[1]) + "|")
            sys.exit()
        
        try:
            PlotECATCorr.ecat_scatter_plot(job_id, clustering_models, df_unlabelled,
                                           cluster_pareto_df, cluster_cols, plot_result_dir)
        except Exception as e:
            print("SPLUNK_LOGGING_MAPA:Failed|" + "At plotting stage: " + str(sys.exc_info()[1]) + "|")
            sys.exit()
        
        if self.output_csv:
            return_data = dict()
            # output clustering results
            return_data, result = self.output_wafer_association(clustering_models, return_data, job_id,
                                                                df_unlabelled, df_labelled, labelled_failure_modes)
            #Cluster Signature json generation
            return_data = self.output_cluster_signature(return_data, job_id, stacked_map_df)
            #Cluster pareto json generation
            return_data = self.output_cluster_pareto(return_data, job_id, cluster_pareto_df)

        if self.output_dataframe:
            return_data["dataframe"] = result

        if self.output_debug_df:
            return_data["raw_data"] = raw_df
        print("SPLUNK_LOGGING_MAPA:Completed|")
        return return_data
    
    def output_wafer_association(self, clustering_models, return_data, job_id,
                                 df_unlabelled, df_labelled, labelled_failure_modes):
        """generate wafer association json"""
        wafer_association_context = ["FAB", "DESIGN_ID", "LOT_ID", "WAFER_ID", "lot_wafer", "DieX", "DieY"]
        result = self.merge_data(wafer_association_context, 
                                 df_unlabelled, df_labelled, 
                                 clustering_models, labelled_failure_modes)
        cluster_result = result.drop_duplicates()
        cluster_result = pd.melt(cluster_result, 
                                 id_vars = wafer_association_context, 
                                 value_vars = clustering_models,
                                 var_name="model", value_name="cluster")
        groupby_cols = ["FAB", "DESIGN_ID", "LOT_ID", "WAFER_ID", "lot_wafer", "model", "cluster"]
        cluster_result = CalStackedWaferMap.aggregate_cluster_die_coordinates(cluster_result, groupby_cols)
        #Cluster wafer-die-coordinate association json generation
        return_data["wafer_association_json"] = str(job_id) + "_" + self.clustering_result_json_file_name
        self.output_json(
            cluster_result,
            os.path.join(self.csv_output_dir, return_data["wafer_association_json"])
        )
        return return_data, result

    def output_cluster_signature(self, return_data, job_id, stacked_map_df):
        """generate cluster signature json"""
        return_data["cluster_signature_json"] = str(job_id) + "_" + self.stacked_map_file_name
        context_col = ['FAB', 'DESIGN_ID', 'cluster', 'model', 'stacked_item']
        cols_tofill = ["num_wafers_in_cluster", "num_dies_in_cluster", "impact",
                        "die_max_value", "die_mean_value", "die_median_value", "die_min_value"]
        stacked_map_df[cols_tofill] = stacked_map_df.groupby(context_col)[cols_tofill]\
            .transform(lambda x: x.fillna(x.mean()))
        aggregated_signatures = CalStackedWaferMap.aggregate_cluster_signature(stacked_map_df, context_col, cols_tofill)
        aggregated_signatures = aggregated_signatures.sort_values(['model','cluster', 'stacked_item'])
        self.output_json(
            aggregated_signatures[context_col + cols_tofill + ["die_info"]], 
            os.path.join(self.csv_output_dir, return_data["cluster_signature_json"])
        )
        return return_data
    
    def output_cluster_pareto(self, return_data, job_id, cluster_pareto_df):
        """generate cluster pareto json"""
        return_data["cluster_esda_pareto_json"] = str(job_id) + "_" + self.cluster_esda_pareto_file_name
        groupby_cols = ['model', 'cluster']
        agg_cols = ['esda_item', 'esda_median', 'cumpercentage']
        wafer_pareto_json = CalStackedWaferMap.aggregate_cluster_pareto(cluster_pareto_df, agg_cols, groupby_cols)
        self.output_json(
            wafer_pareto_json,
            os.path.join(self.csv_output_dir, return_data["cluster_esda_pareto_json"])
        )
        return return_data
    
    def output_json(self, df, output_path):
        """
        output json file to defined output_path
        """
        logger = ESDAKemLogging.getInstance()
        small_case_cols = [e.lower() for e in df.columns]
        df.columns = small_case_cols
        logger.info("Output to JSON file: " + output_path)
        df.to_json(output_path, orient="records")

    def label_die(self, df, labelled_failure_modes, clustering_model):
        for mode in labelled_failure_modes:
            df.loc[df[mode]==1, clustering_model] = mode
        return df
    
    def merge_data(self, wafer_association_context, df_unlabelled, df_labelled, 
                   clustering_models, labelled_failure_modes):
        """merge clustered and labelled data for wafer association output"""
        df_unlabelled = df_unlabelled[wafer_association_context + clustering_models]
        if len(df_labelled)>0:
            df_labelled = df_labelled[wafer_association_context + labelled_failure_modes]
            for col in clustering_models:
                df_labelled = self.label_die(df_labelled, labelled_failure_modes, col)
            result = pd.concat([df_unlabelled, 
                                df_labelled[wafer_association_context + clustering_models]])
        else:
            result = df_unlabelled
        return result

    def __result_to_dict(df):
        pass
