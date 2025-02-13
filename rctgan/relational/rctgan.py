# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:09:24 2022

@author: mohamedg
"""
import os

from rctgan.rdt2 import HyperTransformer
from rctgan.rdt2.transformers import TransformerFactory
from rctgan.tabular import CTGAN, PC_CTGAN
from rctgan.utils import load_yaml_to_dict
from rctgan.utils.dataclass import Config 
from rctgan.utils.enums import FieldType, TransformerType
import pandas as pd
import polars as pl
import numpy as np
import random
import logging

from typing import Dict

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
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def default_hyperparam(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, 'hyperparam.yml')
        
        logging.info(f"Loading default hyperparameters from {yaml_path}")
        default_hyp = load_yaml_to_dict(yaml_path)
        
        if self.hyperparam is None:
            self.hyperparam = {table_name: default_hyp.copy() for table_name in self.metadata.get_tables()}
        else:
            for table_name in self.metadata.get_tables():
                self.hyperparam.setdefault(table_name, default_hyp.copy())
                for hyp, value in default_hyp.items():
                    self.hyperparam[table_name].setdefault(hyp, value)
    
    
    def set_hyperparam(self, hyper):
        self.hyperparam = hyper
        self.default_hyperparam()
    
    def get_hyperparam(self):
        return self.hyperparam
    
    def set_tab_hyperameter(self, table_name, tab_hyper):
        self.hyperparam[table_name] = tab_hyper
        self.default_hyperparam()
    
    def rdt2_transform(self, meta_fields, table, field_deleted=[]):
        ht = HyperTransformer()
        config = Config()
        col_retained = []

        for field, meta in meta_fields.items():
            if field not in field_deleted:
                field_name= meta['type']
                if field_name in FieldType._value2member_map_:
                    field_type = FieldType(field_name)
                    col_retained.append(field)
                    config.sdtypes[field] = field_type.value

                    if field_type == FieldType.DATETIME:
                        format_datetime = meta['format']
                        transformer = TransformerFactory.get_transformer(field_type, format_datetime=format_datetime)
                    else:
                        transformer_type = TransformerType.ONE_HOT_ENCODER if self.ohe_for_parent else TransformerType.FREQUENCY_ENCODER
                        if field_type == FieldType.NUMERICAL:
                            transformer_type = TransformerType.GAUSSIAN_NORMALIZER if self.num_transformers == 'gaussian' else TransformerType.FLOAT_FORMATTER
                        transformer = TransformerFactory.get_transformer(field_type, transformer_type)
                        
                    config.transformers[field] = transformer

        ht.set_config(config=config)
        ht.fit(table[col_retained])
        return ht, col_retained
    
    def gaussian_ht(self, table_transormed):
        ht = HyperTransformer()
        config = Config()
        for col in table_transormed.columns:
            config.sdtypes[col] =  FieldType.NUMERICAL.value
            config.transformers[col] =  TransformerFactory.get_transformer(
                FieldType.NUMERICAL, 
                TransformerType.GAUSSIAN_NORMALIZER
            )
        ht.set_config(config=config)
        ht.fit(table_transormed)
        return ht
    
    def transform(self, table_name: str, table: pl.DataFrame):
        col = self.transformers[table_name]["columns"]
        if not self.if_gaussian_ht:
            return self.transformers[table_name]['hypertr'].transform(table.select(pl.col(col)))
        else:
            transformed_table = self.transformers[table_name]['hypertr'].transform(table.select(pl.col(col)))
            return self.transformers[table_name]['gaussian_ht'].transform(transformed_table)

    def keep_data_col(self, meta_fields):
        return [field for field, meta in meta_fields.items() if meta['type'] in {
            FieldType.CATEGORICAL.value, 
            FieldType.NUMERICAL.value, 
            FieldType.DATETIME.value
            }
        ]
    
    def parents_input_add(self, table_name: str, tables: dict, parents_transformed=None):
        if parents_transformed is None:
            count = 0
        else:
            count = len(parents_transformed.columns)
        parents_name = list(self.metadata.get_parents(table_name))
        for parent_name in parents_name:
            parent_prim_key = self.metadata.get_primary_key(parent_name)
            foreign_keys = list(self.metadata.get_foreign_keys(parent_name, table_name))
    
            for foreign_key in foreign_keys:
                parent_transformed = self.transform(parent_name, tables[parent_name])
                parent_transformed.columns = ["var_" + str(i) for i in range(1, len(parent_transformed.columns) + 1)]
                if self.hyperparam[self.current_table]["grand_parent"]:
                    if parent_name in list(self.metadata.get_parents(self.current_table)):
                        parent_transformed = self.parents_input_add(parent_name, tables, parent_transformed)
                parent_transformed.columns = ["var_" + str(count + i) for i in range(1, len(parent_transformed.columns) + 1)]
                temp_serie = pl.DataFrame({foreign_key: tables[table_name][foreign_key].to_list()})
                parent_transformed = parent_transformed.with_columns(pl.Series(foreign_key, tables[parent_name][parent_prim_key].to_list()))
                parent_transformed = temp_serie.join(parent_transformed, on=foreign_key, how='left')
                parent_transformed = parent_transformed.drop(foreign_key)
    
                if parents_transformed is None:
                    parents_transformed = parent_transformed
                else:
                    parents_transformed = parents_transformed.hstack(parent_transformed)
                count = len(parents_transformed.columns)
                del temp_serie
            del parent_transformed
        return parents_transformed
    
    def process_foreign_keys(self, tables: Dict[str, pl.DataFrame], table_name: str, child_name: str, prim_key: str) -> pl.DataFrame:            
        foreign_keys = list(self.metadata.get_foreign_keys(table_name, child_name))
        for foreign_key in foreign_keys:
            temp_child = tables[child_name].group_by(foreign_key, maintain_order=True).agg(
                pl.count().alias(f"{child_name}_{foreign_key}_nb_occ")
            )
            
            temp_table = tables[table_name].join(temp_child, left_on=prim_key, right_on=foreign_key, how='left')
            temp_table = temp_table.with_columns(
                pl.col(f"{child_name}_{foreign_key}_nb_occ").fill_null(0).cast(pl.Int32)
            )
            
            col_name = f"{child_name}_{foreign_key}_nb_occ"
            
            self.size_stats[table_name][col_name] = {
                "min": temp_child[col_name].min(),
                "max": temp_child[col_name].max(),
                "mean": temp_child[col_name].mean(),
                "std": temp_child[col_name].std(),
            }
        return temp_table
    
    def fit(self, tables : dict):
        table_names = self.metadata.get_tables()
        table_meta = {table_name: self.metadata.get_table_meta(table_name)['fields'] for table_name in table_names}
        table_children = {table_name: list(self.metadata.get_children(table_name)) for table_name in table_names}
        
        for table_name in table_names:
            children = table_children[table_name]
            self.size_tables[table_name] = len(tables[table_name])
            meta = table_meta[table_name]
            
            if children:
                ht, col = self.rdt2_transform(meta, tables[table_name])
                self.transformers[table_name] = {"hypertr": ht, "columns": col}
                if self.if_gaussian_ht:
                    self.transformers[table_name]['gaussian_ht'] = self.gaussian_ht(ht.transform(tables[table_name][col]))
                self.size_stats[table_name] = {}
            else:
                self.transformers[table_name] = {"columns": self.keep_data_col(meta)}
    
        for table_name in table_names:
            self.current_table = table_name
            prim_key = self.metadata.get_primary_key(table_name)
            col_table = self.transformers[table_name]["columns"]
            children = table_children[table_name]
            
            if len(self.metadata.get_parents(table_name)) == 0:
                model = CTGAN(primary_key=prim_key, seed=self.seed, **self.hyperparam[table_name])
                temp_table = tables[table_name].select([prim_key] + col_table).clone()
                for child_name in children:
                    temp_table = self.process_foreign_keys(tables, table_name, child_name, prim_key)
                if self.hyperparam[table_name]["plot_loss"]:
                    print("plot of table: " + table_name)
                model.fit(temp_table)
            else:
                parents_transformed = self.parents_input_add(table_name, tables)
                hyperparams = {k: v for k, v in self.hyperparam[table_name].items() if k != 'grand_parent'}
                model = PC_CTGAN(seed=self.seed, **hyperparams)
                if children:
                    temp_table = tables[table_name].select([prim_key] + col_table).clone()
                    for child_name in children:
                        temp_table = self.process_foreign_keys(tables, table_name, child_name, prim_key)
                    temp_table = temp_table.drop(prim_key)
                else:
                    temp_table = tables[table_name].select(col_table)
                if self.hyperparam[table_name]["plot_loss"]:
                    print("plot of table: " + table_name)
                model.fit(temp_table, parents_transformed)
            
            self.models[table_name] = model
                
    def generate_letter_id(self, size):
        import string
        letters = string.ascii_lowercase
        num_letters = len(letters)
        
        if size <= num_letters:
          return [letters[i] for i in range(size)]
        else:
          result = []
          for i in range(size):
              index = i
              current_id = []
              while index >= 0:
                  current_id.append(letters[index % num_letters])
                  index //= num_letters
              result.append("".join(reversed(current_id)))
          return result     
    
    def parent_child_sample_mini(self, child, sampled_data, table_transformed, f_key_frame):
        sampled_data[child] = self.models[child].sample(list(f_key_frame["_size_"]), table_transformed)

        sampled_data[child] = sampled_data[child].join(f_key_frame, on="Parent_index", how="left")
        sampled_data[child] = sampled_data[child].drop(["_size_", "Parent_index"])

        prim_key = self.metadata.get_primary_key(child)
        if prim_key:
            if self.metadata.get_table_meta(child)['fields'][prim_key]['subtype'] == 'string':
                sampled_data[child] = sampled_data[child].with_column(pl.Series(prim_key, self.generate_letter_id(len(sampled_data[child]))))
            elif self.metadata.get_table_meta(child)['fields'][prim_key]['subtype'] == 'integer':
                sampled_data[child] = sampled_data[child].with_column(pl.Series(prim_key, range(1, len(sampled_data[child]) + 1)))         
    
    def granp_parent_transform_add(self, parent_name, table_transformed, f_key, f_key_frame, sampled_data, tables_transformed):
        grand_parents = list(self.metadata.get_parents(parent_name))
        for grand_parent in grand_parents:
            gp_foreign_keys = list(self.metadata.get_foreign_keys(grand_parent, parent_name))
            for gp_foreign_key in gp_foreign_keys:
                start_var = len(table_transformed.columns)
                temp_serie = pl.DataFrame({gp_foreign_key: sampled_data[parent_name][gp_foreign_key]})
                temp_table = tables_transformed[grand_parent].clone()
                temp_table.columns = ["var_" + str(start_var + i + 1) for i in range(len(temp_table.columns))]
                
                grand_parent_prim_key = self.metadata.get_primary_key(grand_parent)
                temp_table = temp_table.with_column(pl.Series(gp_foreign_key, sampled_data[grand_parent][grand_parent_prim_key]))
                
                temp_serie = temp_serie.join(temp_table, on=gp_foreign_key, how='left')
                temp_serie = temp_serie.drop(gp_foreign_key)
                
                parent_prim_key = self.metadata.get_primary_key(parent_name)
                temp_serie = temp_serie.with_column(pl.Series(f_key, sampled_data[parent_name][parent_prim_key]))
                table_transformed = table_transformed.with_column(pl.Series(f_key, f_key_frame[f_key]))
                table_transformed = table_transformed.join(temp_serie, on=f_key, how='left')
                table_transformed = table_transformed.drop(f_key)
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
                break
            
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

