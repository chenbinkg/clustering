import pandas as pd
import numpy as np

PARETO_NUM = 10

class CalStackedWaferMap(object):
    
    @classmethod
    def generate_stacked_wafer_map(cls, df_labelled, labelled_failure_modes, df_unlabelled, 
                                   clustering_models, probe_item_list, probe_item, bin_step,
                                   esda_items, stacked_items_list):
        """
        Calculate stacked wafer maps for all clusters
        : param df: Pandas DataFrame contains all result.
        : param clustering_models: cluster model names such as kmeans, gmm, l2_gmm
        : param probe_item: column names for probe value such as rdC, rdV, V:V, C:C
        : param esda_items: column names for esda parameters such as ESDA_RDV2_BITR0
        : param stacked_items_list: items to get the stacked wafer map, such as rdC, ESDA_RDV2_BIRT0 etc.
        """
        cls.col_dieX, cls.col_dieY = 'DieX', 'DieY'
        # die_map = df.groupby([col_dieX, col_dieY]).size().reset_index()
        # del die_map[0]
        cls.total_diecount = df_labelled.shape[0] + df_unlabelled.shape[0]
        die_count_per_wfr = df_unlabelled['TOTAL_DIE'].median()
        # wafer_pareto_df = pd.DataFrame(data=None)
        cluster_pareto_df = pd.DataFrame(data=None)
        stacked_map_df = pd.DataFrame(data=None)
        for col, stacked_items in zip(clustering_models, stacked_items_list):
            print('Stacked die level summary for: ', col)
            #Get Stacked Wafer Map for cluster labels
            df2 = cls.get_stacked_map_all(df_unlabelled, col, cls.total_diecount, cls.col_dieX, cls.col_dieY, 
                                          esda_items, stacked_items, probe_item, bin_step, die_count_per_wfr,
                                          probe_item_list)
            df2["model"] = col
            df2.rename(columns={col: "cluster"}, inplace=True)
            #Get Stacked Wafer Map for failure mode labels
            if len(df_labelled)>0:
                #index_col = ["FAB", "DESIGN_ID", "LOT_ID", "WAFER_ID", "lot_wafer", cls.col_dieX, cls.col_dieY] + esda_items
                #df_labelled_s = df_labelled[index_col + labelled_failure_modes]
                index_col = list(set(df_labelled.columns) - set(labelled_failure_modes))
                df_labelled_s = pd.melt(df_labelled, id_vars = index_col, 
                                        value_vars = labelled_failure_modes, 
                                        var_name = 'failure_mode', value_name='failure_mode_val')
                df_labelled_s = df_labelled_s[df_labelled_s['failure_mode_val'] == 1]
                dfx = cls.get_stacked_map_all(df_labelled_s, 'failure_mode', cls.total_diecount, cls.col_dieX, cls.col_dieY, 
                                              esda_items, stacked_items, probe_item, bin_step, die_count_per_wfr,
                                              probe_item_list)
                dfx["model"] = col
                dfx.rename(columns={'failure_mode': "cluster"}, inplace=True)
                df2 = pd.concat([df2, dfx])
            stacked_map_df = pd.concat([stacked_map_df, df2])
            
            #Get cluster signature by Top 10 pareto of ESDA from Stacked Wafer Map
            df1 = df2.groupby(["model", "cluster"])[esda_items].median().reset_index()
            df1 = pd.melt(df1, id_vars=["model", "cluster"], value_vars=esda_items,
                        var_name="esda_item", value_name="esda_median")
            df1 = df1.sort_values(by=["cluster", "esda_median"], ascending=False)
            pareto_num_min = min(PARETO_NUM, len(esda_items))
            df1 = df1.groupby(['model', 'cluster']).apply(lambda x: x.iloc[0:pareto_num_min, ]).reset_index(drop=True)
            df1["cumpercentage"] = df1.groupby(["cluster"])["esda_median"].apply(lambda x: x.cumsum()/x.sum()*100)
            cluster_pareto_df = pd.concat([cluster_pareto_df, df1])
        
        #Create full die map
        die_map = stacked_map_df.groupby(['FAB', 'DESIGN_ID', 'DieX', 'DieY'])['median_value'].median().reset_index()
        del die_map['median_value']
        die_map_final = pd.DataFrame()
        stacked_groupby = stacked_map_df.groupby(['model', 'cluster', 'stacked_item'])
        for group_1 in list(stacked_groupby.groups):
            die_map_1 = die_map.copy()
            die_map_1.loc[:, 'model'] = group_1[0]
            die_map_1.loc[:, 'cluster'] = group_1[1]
            die_map_1.loc[:, 'stacked_item'] = group_1[2]
            die_map_final = die_map_final.append(die_map_1)
        stacked_map_df_all =pd.merge(die_map_final, stacked_map_df, on=['FAB', 'DESIGN_ID', 'DieX', 'DieY',
                                                                        'model', 'cluster', 'stacked_item'], how='left')
        return stacked_map_df_all, cluster_pareto_df
    
    @classmethod
    def get_stacked_map_all(cls, df, col, total_diecount, col_dieX, col_dieY, 
                            esda_items, stacked_items, probe_item, bin_step,
                            die_count_per_wfr, probe_item_list):
        #wafer stacked map for all ESDAs
        groupby_col = ["FAB", "DESIGN_ID", col]
        df2_unique_count = df.groupby(groupby_col)['lot_wafer'].nunique().reset_index()
        df2_unique_count.rename(columns={'lot_wafer': 'num_wafers_in_cluster'}, inplace=True)
        df2_bin_count = df.groupby(groupby_col)[probe_item_list].sum().reset_index()
        df2_bin_count.rename(columns={probe_item_list[0]: 'total_bin_count'}, inplace=True)
        df2_stack_diecount = df.groupby(groupby_col)[stacked_items].count().reset_index()
        df2_stack_diecount = pd.melt(df2_stack_diecount, id_vars = groupby_col, 
                                     value_vars = stacked_items, 
                                     var_name = 'stacked_item', value_name='num_dies_in_cluster')
        df2_stack_max = df.groupby(groupby_col)[stacked_items].max().reset_index()
        df2_stack_max = pd.melt(df2_stack_max, id_vars = groupby_col, 
                                     value_vars = stacked_items, 
                                     var_name = 'stacked_item', value_name='die_max_value')
        df2_stack_min = df.groupby(groupby_col)[stacked_items].min().reset_index()
        df2_stack_min = pd.melt(df2_stack_min, id_vars = groupby_col, 
                                     value_vars = stacked_items, 
                                     var_name = 'stacked_item', value_name='die_min_value')
        df2_stack_mean = df.groupby(groupby_col)[stacked_items].mean().reset_index()
        df2_stack_mean = pd.melt(df2_stack_mean, id_vars = groupby_col, 
                                     value_vars = stacked_items, 
                                     var_name = 'stacked_item', value_name='die_mean_value')
        df2_stack_median = df.groupby(groupby_col)[stacked_items].median().reset_index()
        df2_stack_median = pd.melt(df2_stack_median, id_vars = groupby_col, 
                                     value_vars = stacked_items, 
                                     var_name = 'stacked_item', value_name='die_median_value')
        df2_die_stack = df.groupby([col_dieX, col_dieY] + groupby_col)[stacked_items].median().reset_index()
        df2_die_stack = pd.melt(df2_die_stack, id_vars = [col_dieX, col_dieY] + groupby_col, 
                                value_vars = stacked_items, 
                                var_name = 'stacked_item', value_name='median_value')
        df2_esda_median = df.groupby([col_dieX, col_dieY] + groupby_col)[esda_items].median().reset_index()
        
        df2_stack = pd.merge(df2_stack_diecount, df2_stack_max, on= (groupby_col + ['stacked_item']), how='left')
        df2_stack = pd.merge(df2_stack, df2_stack_min, on= (groupby_col + ['stacked_item']), how='left')
        df2_stack = pd.merge(df2_stack, df2_stack_mean, on= (groupby_col + ['stacked_item']), how='left')
        df2_stack = pd.merge(df2_stack, df2_stack_median, on=(groupby_col + ['stacked_item']), how='left')
        df2_stack = pd.merge(df2_stack, df2_unique_count, on=groupby_col, how='left')
        df2_stack = pd.merge(df2_stack, df2_bin_count, on=groupby_col, how='left')
        df2 = pd.merge(df2_esda_median, df2_stack, on=groupby_col, how='inner')
        df2 = pd.merge(df2, df2_die_stack, on=(groupby_col + [col_dieX, col_dieY, 'stacked_item']), how='left')
        df2["die_loss"] = df2["total_bin_count"]/df2["num_wafers_in_cluster"]/die_count_per_wfr
        df2["impact"] = df2["die_mean_value"]*df2["num_dies_in_cluster"]/total_diecount
        if bin_step == "Fail_bin":
            #replace impact by normalized wafer level yield loss for bin type signature
            df2["impact"] = np.where(df2["stacked_item"] == "BIN_BIT", df2["die_loss"], df2["impact"])
        elif (bin_step != "Fail_bin") and (bin_step is not None):
            df2["impact"] = np.where(df2["stacked_item"] == bin_step, df2["die_loss"], df2["impact"])
        #rename probe_item back to input
        df2["stacked_item"] = np.where(df2["stacked_item"] == probe_item_list[0], probe_item, df2["stacked_item"])
        df2 = df2[
            ["FAB", "DESIGN_ID", col_dieX, col_dieY, col, "stacked_item", "num_wafers_in_cluster", "num_dies_in_cluster",
            "impact", "median_value"] + esda_items + ["die_max_value", "die_mean_value", "die_median_value","die_min_value"]
            ].drop_duplicates()
        return df2

    @classmethod
    def aggregate_cluster_signature(cls, df, context_col, cols_tofill):
        col_dieX, col_dieY = 'DieX', 'DieY'
        col_name = 'die_info'
        df[col_name] = df[[col_dieX, col_dieY, "median_value"]].values.tolist()
        df_item = df.groupby(context_col)[col_name].apply(list).reset_index()
        df_item_stat = df.groupby(context_col)[cols_tofill].mean().reset_index()
        df_item = pd.merge(df_item, df_item_stat[context_col+cols_tofill], on=context_col, how='left')
        return df_item
    
    @classmethod
    def aggregate_cluster_die_coordinates(cls, df, groupby_cols):
        col_dieX, col_dieY = 'DieX', 'DieY'
        col_name = 'die_coordinates'
        df[col_name] = df[[col_dieX, col_dieY]].values.tolist()
        df = df.groupby(groupby_cols)[col_name].apply(list).reset_index()
        df = df.drop_duplicates(subset=groupby_cols)
        df['wafer_die_count'] = df['die_coordinates'].apply(lambda x: len(x))
        df = df.sort_values(["model", "cluster", "wafer_die_count"], ascending=False)
        return df
    
    @classmethod
    def aggregate_cluster_pareto(cls, df, agg_columns, groupby_cols):
        df_agg = pd.DataFrame()
        for item in agg_columns:
            df_item = df.groupby(groupby_cols)[item].apply(list).reset_index()
            if len(df_agg)>0:
                df_agg = pd.merge(df_agg, df_item, on=groupby_cols, how='left')
            else:
                df_agg = df_item
        return df_agg
