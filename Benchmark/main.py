import polars as pl
import time
import os
import pickle
import yaml
import argparse

from rctgan import Metadata
from rctgan.relational import RCTGAN
root_path = os.path.dirname(__file__)

def load_data(pickle_file, sample_fraction=0.1):
    if os.path.exists(os.path.join(root_path, pickle_file)):
        with open(os.path.join(root_path, pickle_file), 'rb') as file:
            tables = pickle.load(file)
    else:
        df_atom = pl.read_csv(os.path.join(root_path, 'atom.csv'))
        df_bond = pl.read_csv(os.path.join(root_path, 'bond.csv'))
        df_molecule = pl.read_csv(os.path.join(root_path, 'molecule.csv'))
        
        df_molecule_sampled = df_molecule.sample(fraction=sample_fraction, seed=42)
        df_atom_sampled = df_atom.filter(pl.col('molecule_id').is_in(df_molecule_sampled['molecule_id']))
        df_bond_sampled = df_bond.filter(
            (pl.col('atom_id').is_in(df_atom_sampled['atom_id'])) |
            (pl.col('atom_id2').is_in(df_atom_sampled['atom_id']))
        )

        tables_name = ['atom', 'bond', 'molecule']
        data_frames = [df_atom_sampled, df_bond_sampled, df_molecule_sampled]
        tables = dict(zip(tables_name, data_frames))

        with open(os.path.join(root_path, pickle_file), 'wb') as file:
            pickle.dump(tables, file)
    
    return tables

def get_metadata(tables):
    metadata = Metadata()

    with open(os.path.join(root_path, 'fields.yml'), 'r') as file:
        metadata_fields = yaml.safe_load(file)

    table_info = [
        {'name': 'atom', 'primary_key': 'atom_id', 'fields': metadata_fields['atom_fields']},
        {'name': 'bond', 'primary_key': None, 'fields': metadata_fields['bond_fields']},
        {'name': 'molecule', 'primary_key': 'molecule_id', 'fields': metadata_fields['molecule_fields']}
    ]

    for info in table_info:
        metadata.add_table(
            name=info['name'],
            data=tables[info['name']],
            primary_key=info['primary_key'],
            fields_metadata=info['fields']
        )

    relationships = [
        {'parent': 'atom', 'child': 'bond', 'foreign_key': 'atom_id'},
        {'parent': 'atom', 'child': 'bond', 'foreign_key': 'atom_id2'},
        {'parent': 'molecule', 'child': 'atom'}
    ]

    for rel in relationships:
        metadata.add_relationship(
            parent=rel['parent'],
            child=rel['child'],
            foreign_key=rel.get('foreign_key')
        )

    return metadata

def count_lines_of_code(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        return len([line for line in lines if line.strip() and not line.strip().startswith('#')])

def count_lines_of_code_in_folder(folder_path):
    total_lines = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                total_lines += count_lines_of_code(os.path.join(root, file))
    return total_lines

def main():
    parser = argparse.ArgumentParser(description='Benchmark RCTGAN')
    parser.add_argument('--pickle_file', type=str, required=False, default='benchmark_data.pkl', help='Path to the pickle file for data')
    parser.add_argument('--csv_file', type=str, required=False, default='benchmark_results.csv', help='Path to the CSV file for results')
    parser.add_argument('--sample_fraction', type=float, required=False, default=0.1, help='Fraction of data to sample for the benchmark')
    parser.add_argument('--epochs', type=int, required=False, default=3, help='Number of epochs for benchmark')
    args = parser.parse_args()

    tables = load_data(args.pickle_file, args.sample_fraction)
    metadata = get_metadata(tables)

    hyper = {
        "verbose": False,
        "epochs": args.epochs,
    }
    
    lines_of_code = count_lines_of_code_in_folder('/home/lab/RCTGAN/rctgan')
    print(f'Nombre de lignes de code dans le dossier RCTGAN: {lines_of_code}')

    start_time = time.time()

    model = RCTGAN(metadata=metadata, hyperparam=hyper)
    model.fit(tables)

    end_time = time.time()
    execution_time = end_time - start_time

    if os.path.exists(os.path.join(root_path, args.csv_file)):
        results_df = pl.read_csv(os.path.join(root_path, args.csv_file))
    else:
        results_df = pl.DataFrame(columns=['version', 'execution_time', 'lines_of_code'])

    if not ((results_df['execution_time'] == execution_time) & (results_df['lines_of_code'] == lines_of_code)).any():
        new_entry = pl.DataFrame([{
            'version': results_df.shape[0] + 1,
            'execution_time': execution_time,
            'lines_of_code': lines_of_code,
            'epochs': args.epochs
        }])
        results_df = pl.concat([results_df, new_entry])
        results_df.write_csv(os.path.join(root_path, args.csv_file))

    print(results_df)

if __name__ == '__main__':
    main()