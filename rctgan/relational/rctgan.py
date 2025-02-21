# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:09:24 2022

@author: mohamedg
"""
from rctgan.rdt2 import HyperTransformer
from rctgan.rdt2.transformers.numerical import FloatFormatter, GaussianNormalizer
from rctgan.rdt2.transformers.categorical import FrequencyEncoder, OneHotEncoder
from rctgan.rdt2.transformers.datetime import OptimizedTimestampEncoder
import itertools
from rctgan.tabular import CTGAN, PC_CTGAN
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm, kstest
import sys

class RCTGAN:
    def __init__(self, metadata=None, hyperparam=None, current_table=None,
                 ohe_for_parent=False, if_gaussian_ht=True, num_transformers='gaussian', seed=None):
        self.metadata = metadata
        self.transformers = {}
        self.size_tables = {}
        self.size_stats = {}
        self.models = {}
        self.hyperparam = hyperparam
        self.default_hyperparam()
        self.current_table = current_table
        self.ohe_for_parent = ohe_for_parent
        self.if_gaussian_ht = if_gaussian_ht
        if self.ohe_for_parent:
            self.if_gaussian_ht = False
        if self.if_gaussian_ht==False:
            self.num_transformers = num_transformers
        elif num_transformers=='gaussian':
            self.num_transformers = 'float'
        else:
            self.num_transformers = num_transformers
        self.seed = seed
    
    def default_hyperparam(self):
        default_hyp = {"embedding_dim": 128,
                       "generator_dim": (256, 256),
                       "discriminator_dim": (256, 256),
                       "generator_lr": 2e-4,
                       "generator_decay": 1e-6,
                       "discriminator_lr": 2e-4,
                       "discriminator_decay": 1e-6,
                       "batch_size": 500,
                       "discriminator_steps": 1,
                       "log_frequency": True,
                       "verbose": False,
                       "epochs": 1000,
                       "pac": 10,
                       "cuda": True,
                       "plot_loss": False,
                       "grand_parent": True,
                       "field_transformers": None,
                       "anonymize_fields": None,
                       "constraints": None,
                       "table_metadata": None,
                       "rounding": 'auto', 
                       'min_value': 'auto', 
                       'max_value': 'auto'
                      }
        if self.hyperparam==None:
            self.hyperparam = {}
            for table_name in list(self.metadata.get_tables()):
                self.hyperparam[table_name] = default_hyp
        else:
            for table_name in list(self.metadata.get_tables()):
                if not table_name in list(self.hyperparam.keys()):
                    self.hyperparam[table_name] = default_hyp
                else:
                    for hyp in list(default_hyp.keys()):
                        if not hyp in list(self.hyperparam[table_name].keys()):
                            self.hyperparam[table_name][hyp] = default_hyp[hyp]
    
    
    def set_hyperparam(self, hyper):
        self.hyperparam = hyper
        self.default_hyperparam()
    
    def get_hyperparam(self):
        return self.hyperparam
    
    def set_tab_hyperameter(self, table_name, tab_hyper):
        self.hyperparam[table_name] = tab_hyper
        self.default_hyperparam()
    
    def rdt2_transform(self, meta_fields, table, field_deleted=[]): # meta_fields = metadata.get_table_meta('store')['fields']
        ht = HyperTransformer()
        config_dict = {"sdtypes": {}, "transformers": {}}
        col_retained = []
        for field in meta_fields.keys():
            if field not in field_deleted:
                if meta_fields[field]['type'] == 'categorical':
                    col_retained.append(field)
                    config_dict["sdtypes"][field] =  'categorical'
                    if self.ohe_for_parent:
                        config_dict["transformers"][field] =  OneHotEncoder()
                    else:
                        config_dict["transformers"][field] =  FrequencyEncoder(add_noise=True)
                elif meta_fields[field]['type'] == 'numerical':
                    col_retained.append(field)
                    config_dict["sdtypes"][field] =  'numerical'
                    if self.num_transformers=='gaussian':
                        config_dict["transformers"][field] =  GaussianNormalizer()
                    elif self.num_transformers=='float':
                        config_dict["transformers"][field] =  FloatFormatter(missing_value_replacement='mean')
                    else:
                        config_dict["transformers"][field] =  GaussianNormalizer()
                    
                elif meta_fields[field]['type'] == 'datetime':
                    col_retained.append(field)
                    config_dict["sdtypes"][field] =  'datetime'
                    format_datetime = meta_fields[field]['format']
                    config_dict["transformers"][field] =  OptimizedTimestampEncoder(missing_value_replacement='mean', datetime_format=format_datetime)
        ht.set_config(config=config_dict)
        ht.fit(table[col_retained])
        return ht, col_retained
    
    def gaussian_ht(self, table_transormed):
        ht = HyperTransformer()
        config_dict = {"sdtypes": {}, "transformers": {}}
        for col in table_transormed.columns:
            config_dict["sdtypes"][col] =  'numerical'
            config_dict["transformers"][col] =  GaussianNormalizer()
        ht.set_config(config=config_dict)
        ht.fit(table_transormed)
        return ht
    
    def transform(self, table_name, table):
        col = self.transformers[table_name]["columns"]
        if self.if_gaussian_ht==False:
            return self.transformers[table_name]['hypertr'].transform(table[col])
        else:
            return  self.transformers[table_name]['gaussian_ht'].transform(self.transformers[table_name]['hypertr'].transform(table[col]))

    def keep_data_col(self, meta_fields):
        col_retained = []
        for field in meta_fields.keys():
            if meta_fields[field]['type'] == 'categorical':
                col_retained.append(field)
            elif meta_fields[field]['type'] == 'numerical':
                col_retained.append(field)
            elif meta_fields[field]['type'] == 'datetime':
                col_retained.append(field)
        return col_retained
    
    def parents_input_add(self, table_name, tables, parents_trandformed=None):
        
        if str(parents_trandformed)=="None":
            count = 0
        else:
            count = len(parents_trandformed.columns)
        parents_name = list(self.metadata.get_parents(table_name))
        for parent_name in parents_name:
            parent_prim_key = self.metadata.get_primary_key(parent_name)
            foreign_keys = list(self.metadata.get_foreign_keys(parent_name, table_name))
            bool_test = False
            
            ht = self.transformers[parent_name]["hypertr"]
            col = self.transformers[parent_name]["columns"]

            for foreign_key in foreign_keys:
                parent_trandformed = self.transform(parent_name, tables[parent_name])
                parent_trandformed.columns = ["var_"+str(i) for i in range(1, len(parent_trandformed.columns)+1)]
                if self.hyperparam[self.current_table]["grand_parent"]:
                    if parent_name in list(self.metadata.get_parents(self.current_table)):
                        parent_trandformed = self.parents_input_add(parent_name, tables, parent_trandformed)
                parent_trandformed.columns = ["var_"+str(count+i) for i in range(1, len(parent_trandformed.columns)+1)]
                temp_serie = pd.DataFrame(tables[table_name][foreign_key].copy())
                parent_trandformed[foreign_key] = list(tables[parent_name][parent_prim_key])
                parent_trandformed = temp_serie.merge(parent_trandformed, 
                                                              on=[foreign_key],
                                                              how='left', 
                                                              indicator=True)
                parent_trandformed = parent_trandformed.drop([foreign_key, '_merge'], axis=1)
                
                if str(parents_trandformed)=="None":
                    parents_trandformed = parent_trandformed.copy()
                else:
                    parents_trandformed = parents_trandformed.join(parent_trandformed)
                count = len(parents_trandformed.columns)
                del temp_serie
            del parent_trandformed
        return parents_trandformed
    
    def fit(self, tables):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        for table_name in self.metadata.get_tables():
            children = list(self.metadata.get_children(table_name))
            self.size_tables[table_name] =len(tables[table_name])
            if len(children)>0:
                meta = self.metadata.get_table_meta(table_name)['fields']
                ht, col = self.rdt2_transform(meta, tables[table_name])
                self.transformers[table_name] = {"hypertr": ht, "columns": col}
                del ht
                del col
                if self.if_gaussian_ht:
                    ht = self.transformers[table_name]['hypertr']
                    col = self.transformers[table_name]['columns']
                    self.transformers[table_name]['gaussian_ht'] = self.gaussian_ht(ht.transform(tables[table_name][col]))

                self.size_stats[table_name] = {}
            else:
                meta = self.metadata.get_table_meta(table_name)['fields']
                self.transformers[table_name] = {"columns": self.keep_data_col(meta)}

        for table_name in self.metadata.get_tables():
            self.current_table = table_name
            if len(self.metadata.get_parents(table_name)) == 0:
                prim_key = self.metadata.get_primary_key(table_name)
                model = CTGAN(primary_key=prim_key, 
                              embedding_dim=self.hyperparam[table_name]["embedding_dim"], 
                              generator_dim=self.hyperparam[table_name]["generator_dim"], 
                              discriminator_dim=self.hyperparam[table_name]["discriminator_dim"],
                              generator_lr=self.hyperparam[table_name]["generator_lr"], 
                              generator_decay=self.hyperparam[table_name]["generator_decay"], 
                              discriminator_lr=self.hyperparam[table_name]["discriminator_lr"],
                              discriminator_decay=self.hyperparam[table_name]["discriminator_decay"], 
                              batch_size=self.hyperparam[table_name]["batch_size"], 
                              discriminator_steps=self.hyperparam[table_name]["discriminator_steps"],
                              log_frequency=self.hyperparam[table_name]["log_frequency"], 
                              verbose=self.hyperparam[table_name]["verbose"], 
                              epochs=self.hyperparam[table_name]["epochs"], 
                              pac=self.hyperparam[table_name]["pac"], 
                              cuda=self.hyperparam[table_name]["cuda"],
                              plot_loss=self.hyperparam[table_name]["plot_loss"],
                              seed=self.seed,
                              field_transformers=self.hyperparam[table_name]["field_transformers"],
                              anonymize_fields=self.hyperparam[table_name]["anonymize_fields"],
                              constraints=self.hyperparam[table_name]["constraints"],
                              table_metadata=self.hyperparam[table_name]["table_metadata"],
                              rounding=self.hyperparam[table_name]["rounding"],
                              min_value=self.hyperparam[table_name]["min_value"],
                              max_value=self.hyperparam[table_name]["max_value"])
                col_table = self.transformers[table_name]["columns"]
                children = list(self.metadata.get_children(table_name))
                temp_table = tables[table_name][[prim_key]+col_table].copy()
                for child_name in children:
                    '''
                    Create the number of occurrence columns for the children of table_name.
                    Those columns count the number of children of each row for table_name.
                    they take part of the modeling and  will be usefull during sampling step for
                    chosing the number of children to generate of each sampled row in table_name.
                    '''
                    foreign_keys = list(self.metadata.get_foreign_keys(table_name, child_name))
                    for foreign_key in foreign_keys:
                        temp_child = pd.DataFrame(tables[child_name].groupby([foreign_key])[foreign_key].count())
                        temp_child.columns = [child_name+"_"+foreign_key+"_nb_occ"]
                        temp_child[prim_key] = temp_child.index
                        temp_child.index = range(len(temp_child))
                        temp_table = temp_table.merge(temp_child, 
                                                      on=[prim_key], 
                                                      how='left', 
                                                      indicator=True)
                        temp_table = temp_table.drop(['_merge'], axis=1)
                        mask = temp_table[child_name+"_"+foreign_key+"_nb_occ"].isna()
                        temp_table.loc[mask, child_name+"_"+foreign_key+"_nb_occ"] = 0
                        temp_table[child_name+"_"+foreign_key+"_nb_occ"] = temp_table[child_name+"_"+foreign_key+"_nb_occ"].astype(int)
                        self.size_stats[table_name][child_name+"_"+foreign_key+"_nb_occ"] = {"min": np.min(temp_table[child_name+"_"+foreign_key+"_nb_occ"]),
                                                                                             "max": np.max(temp_table[child_name+"_"+foreign_key+"_nb_occ"]),
                                                                                             "mean": np.mean(temp_table[child_name+"_"+foreign_key+"_nb_occ"]),
                                                                                             "std": np.std(temp_table[child_name+"_"+foreign_key+"_nb_occ"])
                                                                                            }
                if self.hyperparam[table_name]["plot_loss"]==True:
                    print("plot of table: "+table_name)

                model.fit(temp_table)
                self.models[table_name] = model
                del temp_table
            
            else:
                prim_key = self.metadata.get_primary_key(table_name)
                parents_trandformed = self.parents_input_add(table_name, tables)
                model = PC_CTGAN(embedding_dim=self.hyperparam[table_name]["embedding_dim"], 
                                 generator_dim=self.hyperparam[table_name]["generator_dim"], 
                                 discriminator_dim=self.hyperparam[table_name]["discriminator_dim"],
                                 generator_lr=self.hyperparam[table_name]["generator_lr"], 
                                 generator_decay=self.hyperparam[table_name]["generator_decay"], 
                                 discriminator_lr=self.hyperparam[table_name]["discriminator_lr"],
                                 discriminator_decay=self.hyperparam[table_name]["discriminator_decay"], 
                                 batch_size=self.hyperparam[table_name]["batch_size"], 
                                 discriminator_steps=self.hyperparam[table_name]["discriminator_steps"],
                                 log_frequency=self.hyperparam[table_name]["log_frequency"], 
                                 verbose=self.hyperparam[table_name]["verbose"], 
                                 epochs=self.hyperparam[table_name]["epochs"], 
                                 pac=self.hyperparam[table_name]["pac"], 
                                 cuda=self.hyperparam[table_name]["cuda"],
                                 plot_loss=self.hyperparam[table_name]["plot_loss"],
                                 seed=self.seed,
                                 field_transformers=self.hyperparam[table_name]["field_transformers"],
                                anonymize_fields=self.hyperparam[table_name]["anonymize_fields"],
                                constraints=self.hyperparam[table_name]["constraints"],
                                table_metadata=self.hyperparam[table_name]["table_metadata"],
                                rounding=self.hyperparam[table_name]["rounding"],
                                min_value=self.hyperparam[table_name]["min_value"],
                                max_value=self.hyperparam[table_name]["max_value"])
                    
                col_table = self.transformers[table_name]["columns"]
                children = list(self.metadata.get_children(table_name))
                if len(children)>0:
                    '''
                    Create the number of occurrence columns for the children of table_name.
                    Those columns count the number of children of each row for table_name.
                    they take part of the modeling and  will be usefull during sampling step for
                    chosing the number of children to generate of each sampled row in table_name.
                    '''
                    temp_table = tables[table_name][[prim_key]+col_table].copy()
                    for child_name in children:
                        foreign_keys = list(self.metadata.get_foreign_keys(table_name, child_name))
                        for foreign_key in foreign_keys:
                            temp_child = pd.DataFrame(tables[child_name].groupby([foreign_key])[foreign_key].count())
                            temp_child.columns = [child_name+"_"+foreign_key+"_nb_occ"]
                            temp_child[prim_key] = temp_child.index
                            temp_child.index = range(len(temp_child))
                            temp_table = temp_table.merge(temp_child, 
                                                          on=[prim_key], 
                                                          how='left', 
                                                          indicator=True)
                            temp_table = temp_table.drop(['_merge'], axis=1)
                            mask = temp_table[child_name+"_"+foreign_key+"_nb_occ"].isna()
                            temp_table.loc[mask, child_name+"_"+foreign_key+"_nb_occ"] = 0
                            temp_table[child_name+"_"+foreign_key+"_nb_occ"] = temp_table[child_name+"_"+foreign_key+"_nb_occ"].astype(int)
                            self.size_stats[table_name][child_name+"_"+foreign_key+"_nb_occ"] = {"min": np.min(temp_table[child_name+"_"+foreign_key+"_nb_occ"]),
                                                                                             "max": np.max(temp_table[child_name+"_"+foreign_key+"_nb_occ"]),
                                                                                             "mean": np.mean(temp_table[child_name+"_"+foreign_key+"_nb_occ"]),
                                                                                             "std": np.std(temp_table[child_name+"_"+foreign_key+"_nb_occ"])
                                                                                            }
                    temp_table = temp_table.drop([prim_key], axis=1)
                else:
                    temp_table = tables[table_name][col_table]
                if self.hyperparam[table_name]["plot_loss"]==True:
                    print("plot of table: "+table_name)

                model.fit(temp_table, parents_trandformed)
                self.models[table_name] = model
                
    def generate_letter_id(self, size):
        liste = []
        # removed n as nan ids were being generated
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
                   'j', 'k', 'l', 'm', 'o', 'p', 'q', 'r',
                   's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        is_first = True
        boolean =True
        while boolean:
            if is_first:
                combin_letters = letters
                liste += combin_letters
                is_first = False
            else:
                combin_letters_list = list(itertools.product(combin_letters, letters))
                combin_letters = []
                for c in combin_letters_list:
                    combin_letters += [c[0]+c[1]]
                liste += combin_letters
                
            if len(liste) > size:
                liste = liste[:size]
                boolean = False
        return liste
            
        
    
    def parent_child_sample_mini(self, child, sampled_data, table_transformed, f_key_frame):
        sampled_data[child] = self.models[child].sample(list(f_key_frame["_size_"]), table_transformed)

        sampled_data[child] = sampled_data[child].merge(f_key_frame, 
                                                          on=["Parent_index"],
                                                          how='left', 
                                                          indicator=True)
        sampled_data[child] = sampled_data[child].drop(["_size_", "Parent_index", "_merge"], axis=1)

        prim_key = self.metadata.get_primary_key(child)
        if prim_key:
            if self.metadata.get_table_meta(child)['fields'][prim_key]['subtype']=='string':
                sampled_data[child][prim_key] = self.generate_letter_id(len(sampled_data[child]))
            elif self.metadata.get_table_meta(child)['fields'][prim_key]['subtype']=='integer':
                sampled_data[child][prim_key] = range(1, len(sampled_data[child])+1)
        
    
    def granp_parent_transform_add(self, parent_name, table_transformed, f_key, f_key_frame, sampled_data, tables_transformed):
        grand_parents = list(self.metadata.get_parents(parent_name))
        for grand_parent in grand_parents:
            gp_foreign_keys = list(self.metadata.get_foreign_keys(grand_parent, parent_name))
            for gp_foreign_key in gp_foreign_keys:
                start_var = len(table_transformed.columns)
                temp_serie = pd.DataFrame(sampled_data[parent_name][gp_foreign_key])
                temp_serie.columns = [gp_foreign_key]
                temp_table = tables_transformed[grand_parent].copy()
                temp_table.columns = ["var_"+str(start_var+i+1) for i in range(len(temp_table.columns))] 
                
                grand_parent_prim_key = self.metadata.get_primary_key(grand_parent)
                temp_table[gp_foreign_key] = sampled_data[grand_parent][grand_parent_prim_key]
                
                temp_serie = temp_serie.merge(temp_table, on=[gp_foreign_key], how='left', indicator=True)
                temp_serie = temp_serie.drop([gp_foreign_key, "_merge"], axis=1)
                
                parent_prim_key = self.metadata.get_primary_key(parent_name)
                temp_serie[f_key] = sampled_data[parent_name][parent_prim_key]
                table_transformed[f_key] = f_key_frame[f_key]
                table_transformed = table_transformed.merge(temp_serie, on=[f_key], how='left', indicator=True)
                table_transformed = table_transformed.drop([f_key, "_merge"], axis=1)
        return table_transformed
                
    def dupli_rows(self, data, size_list):
        if len(data)==len(size_list):
            df = data.copy()
            df.index = range(len(df))
            df['index_prim_key'] = range(len(df))
            size_list_2 = []
            for k in range(len(size_list)):
                s = size_list[k]
                size_list_2 += [k]*s
            index_prim_key = pd.DataFrame(size_list_2, columns=['index_prim_key'])
            df = index_prim_key.merge(df, on=['index_prim_key'], how='left')
            df = df.drop(['index_prim_key'], axis=1)
            return df
        else:
            return None


    def parent_child_sample(self, child, sampled_data, tables_transformed):
        parents_name = list(self.metadata.get_parents(child))
        all_parents_sampled = True
        for parent_name in parents_name:
            if not parent_name in list(sampled_data.keys()):
                all_parents_sampled = False
            else:
                if self.hyperparam[child]["grand_parent"]:
                    grand_parents = list(self.metadata.get_parents(parent_name))
                    for grand_parent in grand_parents:
                        if not grand_parent in list(sampled_data.keys()):
                            all_parents_sampled = False
                    
            if all_parents_sampled == False:
                # If all parents were not sampled yet, exit the function
                # the table will be sampled when all its parents are sampled.
                return
            
            if parent_name not in tables_transformed.keys():
                table_transformed = self.transform(parent_name, sampled_data[parent_name])
                tables_transformed[parent_name] = table_transformed.copy()
                del table_transformed
        
        if all_parents_sampled:
            enc_parent = parents_name[0]
            enc_foreign_key = list(self.metadata.get_foreign_keys(enc_parent, child))[0]
            enc_nb_occ_name = child+"_"+enc_foreign_key+"_nb_occ"
            
            table_transformed = tables_transformed[enc_parent].copy()
            table_transformed.columns = ["var_"+str(i+1) for i in range(len(table_transformed.columns))]
            
            prim_enc = self.metadata.get_primary_key(enc_parent)
            f_key_frame = pd.DataFrame(sampled_data[enc_parent][[prim_enc, enc_nb_occ_name]])
            f_key_frame.columns = [enc_foreign_key, "_size_"]
            f_key_frame["Parent_index"] = list(f_key_frame.index)
            
            if self.hyperparam[child]["grand_parent"]:
                table_transformed = self.granp_parent_transform_add(enc_parent, table_transformed, enc_foreign_key, f_key_frame, sampled_data, tables_transformed)
            
            if len(parents_name)==1 and len(list(self.metadata.get_foreign_keys(enc_parent, child)))==1:
                self.parent_child_sample_mini(child, sampled_data, table_transformed, f_key_frame)
            else:
                table_transformed = self.dupli_rows(table_transformed, list(f_key_frame["_size_"]))
                f_key_frame = self.dupli_rows(f_key_frame, list(f_key_frame["_size_"]))
                f_key_frame["_size_"] = [1 for _ in range(len(f_key_frame))]
                f_key_frame["Parent_index"] = list(f_key_frame.index)

                for parent_name in parents_name:
                    foreign_keys = list(self.metadata.get_foreign_keys(parent_name, child))
                    if parent_name==enc_parent:
                        foreign_keys.remove(enc_foreign_key)
                    for foreign_key in foreign_keys:
                        start_var = len(table_transformed.columns)
                        indexes_chosen = list(random.choices(tables_transformed[parent_name].index, k=len(table_transformed)))
                        random_parent_rows_transformed = tables_transformed[parent_name].loc[indexes_chosen].copy()
                        random_parent_rows_transformed.columns = ["var_"+str(start_var+i+1) for i in range(len(random_parent_rows_transformed.columns))]
                        random_parent_rows_transformed.index = range(len(random_parent_rows_transformed))
                        table_transformed = pd.concat([table_transformed, random_parent_rows_transformed], axis=1)
                        del random_parent_rows_transformed
                        
                        prim_key = self.metadata.get_primary_key(parent_name)
                        f_key_frame[foreign_key] = list(sampled_data[parent_name].loc[indexes_chosen][prim_key])
                        if self.hyperparam[child]["grand_parent"]:
                            table_transformed = self.granp_parent_transform_add(parent_name, table_transformed, foreign_key, f_key_frame, sampled_data, tables_transformed)
                self.parent_child_sample_mini(child, sampled_data, table_transformed, f_key_frame)
            
        children = list(self.metadata.get_children(child))
        if len(children)>0:
            table_transformed = self.transform(child, sampled_data[child])
            tables_transformed[child] = table_transformed.copy()
            del table_transformed
        for child_2 in children:
            self.parent_child_sample(child_2, sampled_data, tables_transformed)
            
        
    def sample(self):
        sampled_data = {}
        tables_transformed = {}
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        for table_name in self.metadata.get_tables():
            if len(self.metadata.get_parents(table_name)) == 0:
                prim_key = self.metadata.get_primary_key(table_name)
                n_table = self.size_tables[table_name]
                sampled_data[table_name] = self.models[table_name].sample(n_table)
                children = list(self.metadata.get_children(table_name))
                if len(children)>0:
                    table_transformed = self.transform(table_name, sampled_data[table_name])
                    tables_transformed[table_name] = table_transformed.copy()
                    del table_transformed
                for child in children:
                    self.parent_child_sample(child, sampled_data, tables_transformed)
        
        for tab_name in sampled_data.keys():
            for c in sampled_data[tab_name].columns:
                if c[-7:] == '_nb_occ':
                    sampled_data[tab_name] = sampled_data[tab_name].drop([c], axis=1)
        return sampled_data

