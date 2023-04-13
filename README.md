<div align="center">
<br/>
<p align="center">
    <i>This repository is about <a href="https://arxiv.org/abs/2211.07588">Row Condition Tabular GAN</a>, a project from <a href="https://www.croesus.com/about-us/croesus-lab/">DataCebo</a>.</i>
</p>

<div align="left">
<br/>
<p align="center">
<a href="https://arxiv.org/abs/2211.07588">
<img align="center" width=40% src="/rctgan/images/logo_rctgan.PNG"></img>
</a>
</p>
</div>

</div>

# Overview

The **Row Conditional Tabular GAN (RC-TGAN)** is the first method for generating synthetic relational databases based on GAN in our knowledge. The RC-TGAN models relationship information between tables by incorporating conditional data of parent rows into the design of the child table's GAN. We further extend the RC-TGAN to model the influence that grandparent table rows may have on their grandchild rows, in order to prevent the loss of this connection when the rows of the parent table fail to transfer this relationship information. For more details see our article on arxiv: <a href="https://arxiv.org/abs/2211.07588">Row Conditional-TGAN for Generating Synthetic Relational Databases</a>.

This repository is the implementation of RC-TGAN and is based on <a href="https://github.com/sdv-dev">
The Synthetic Data Vault Project</a> repositories.

# Install

**Using `pip`:**

```bash
pip install RCTGAN
```


For more installation options please visit the [SDV installation Guide](
https://sdv.dev/SDV/getting_started/install.html)

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started using **RCTGAN**.

## 1. Model the dataset using RCTGAN

To model a multi table, relational dataset, we follow two steps. In the first step, we will load
the data and configures the meta data. In the second step, we will use the sdv API to fit and
save a hierarchical model. We will cover these two steps in this section using an example dataset.

### Step 1: Load dataset and define Metadata
The dataset we used is: <a href="https://github.com/sdv-dev">lien</a>.

```python3
import pandas as pd
from rctgan import Metadata
from rctgan.relational import RCTGAN

df_atom = pd.read_csv('atom.csv')
df_bond = pd.read_csv('bond.csv')
df_molecule = pd.read_csv('molecule.csv')
```

Let's transform dataframes to a dictionary of dataframes and define Metadata. For more details about Metadata see: [Working with Metadata](https://sdv.dev/SDV/user_guides/relational/relational_metadata.html)
tutorial.

```python
tables_name = ['atom', 'bond', 'molecule']
data_frames = [df_atom, df_bond, df_molecule]

tables = dict(zip(tables_name, data_frames))
```

The returned objects contain the following information:

```
{'atom':               atom_id molecule_id type
 0       i100_02_7_10i  i100_02_7i    c
 1     i100_02_7_10_1i  i100_02_7i    h
 2        i100_02_7_1i  i100_02_7i    o
 ...               ...         ...  ...
 6566      i99_65_0_8i   i99_65_0i    c
 6567      i99_65_0_9i   i99_65_0i    n
 
 [6568 rows x 3 columns],
 'bond':             atom_id         atom_id2  type
 0     i100_02_7_10i  i100_02_7_10_1i     1
 1      i100_02_7_1i     i100_02_7_2i     2
 ...             ...              ...   ...
 6614    i99_65_0_9i     i99_65_0_10i     2
 6615    i99_65_0_9i     i99_65_0_11i     2
 
 [6616 rows x 3 columns],
 'molecule':     molecule_id  activity  logp  mweight
 0    i100_02_7i   4.53367  1.91  139.110
 1    i100_21_0i   4.56435  1.76  166.131
 ..          ...       ...   ...      ...
 326   i99_59_2i   5.85220  1.55  168.151
 327   i99_65_0i   7.82244  1.63  168.108
  
 [328 rows x 4 columns]}
```

Let's define Metadata.

```python
#creation de l'instance matadata
metadata = Metadata()

#Specification des proprietees des differents champs

atom_fields = {
    'atom_id': {
        'type': 'id',
        'subtype': 'string'
    },
    'molecule_id': {
        'type': 'id',
        'subtype': 'string'
    },
    'type': {
        'type': 'categorical'
    }   
 }

bond_fields = {
    'atom_id': {
        'type': 'id',
        'subtype': 'string'
    },
    'atom_id2': {
        'type': 'id',
        'subtype': 'string'
    },
    'type': {
        'type': 'categorical'
    }
 }

molecule_fields = {
    'molecule_id': {
        'type': 'id',
        'subtype': 'string'
    },
    'activity': {
        'type': 'numerical',
        'subtype': 'float'
    },
    'logp': {
        'type': 'numerical',
        'subtype': 'float'
    },
    'mweight': {
        'type': 'numerical',
        'subtype': 'float'
    },
    
 }

#Ajout des tables 

metadata.add_table(
     name='atom',
     data=tables['atom'],
     primary_key='atom_id',
     fields_metadata = atom_fields
 )

metadata.add_table(
     name='bond',
     data=tables['bond'],
     fields_metadata = bond_fields
 )

metadata.add_table(
     name='molecule',
     data=tables['molecule'],
     primary_key='molecule_id',
     fields_metadata = molecule_fields
 )

#Ajout des relations
metadata.add_relationship(parent='atom', child='bond', foreign_key = 'atom_id')
metadata.add_relationship(parent='atom', child='bond', foreign_key = 'atom_id2')
metadata.add_relationship(parent='molecule', child='atom')
```

### 2. Fit a model using the RCTGAN API.

First, we build a hierarchical statistical model of the data using **RCTGAN**. For this we will
create an instance of the `rctgan.RCTGAN` class and use its `fit` method.

During this process, **RCTGAN** will traverse across all the tables in your dataset following the
primary key-foreign key relationships and learn the probability distributions of the values in
the columns.

```python3
from rctgan import RCTGAN

model = RCTGAN(metadata)
model.fit(tables)
```

You can save the model with pickle.

```python3
import pickle
pickle.dump(model, open('model_rctgan.p', "wb" ) )
```

The generated `pkl` file will not include any of the original data in it, so it can be
safely sent to where the synthetic data will be generated without any privacy concerns.

## 2. Sample data from the fitted model

In order to sample data from the fitted model, we will first need to load it from its
`p` file. Note that you can skip this step if you are running all the steps sequentially
within the same python session.

```python3
model = pickle.load(open("model_rctgan.p", "rb" ) )
```

After loading the instance, we can sample synthetic data by calling its `sample` method.

```python3
new_data = model.sample()
```

The output will be a dictionary with the same structure as the original `tables` dict,
but filled with synthetic data instead of the real one.

Finally, if you want to evaluate how similar the sampled tables are to the real data,
please have a look at our [evaluation](EVALUATION.md) framework or visit the [SDMetrics](
https://github.com/sdv-dev/SDMetrics) library.

## 3. Hyperparameters configuration
Each table is modeled by a modified CTGAN. In RCTGAN, we can tune the hyperparameters of each CTGAN (tables) through a dictionnary.

```python3
hyper = {'molecule': {'embedding_dim':64,
                      'generator_lr': 2e-5,
                      'generator_dim': (256, 256)
                     },
         'atom': {'embedding_dim':12,
                  'generator_lr': 2e-4,
                  'generator_dim': (128, 128),
                  'batch_size': 10000
                 },
         'bond': {'embedding_dim':12,
                  'generator_lr': 2e-4,
                  'generator_dim': (64, 64),
                  'batch_size': 10000,
                  'grand_parent': True
                 }
        }
model = RCTGAN(metadata, hyper)
model.fit(tables)
```

The following table overview and describe hyperparameters:
| Hyparameters Description | |
| --------------------------------------------- | -------------------------------------------------------------------- |
embedding_dim (int) | Size of the random sample passed to the Generator. Defaults to 128|
generator_dim (tuple or list of ints) | Size of the output samples for each one of the Residuals. A Residual Layer will be created for each one of the values provided. Defaults to (256, 256)|
discriminator_dim (tuple or list of ints) | Size of the output samples for each one of the Discriminator Layers. A Linear Layer will be created for each one of the values provided. Defaults to (256, 256)|
generator_lr (float) | Learning rate for the generator. Defaults to 2e-4|
generator_decay (float) | Generator weight decay for the Adam Optimizer. Defaults to 1e-6|
discriminator_lr (float) | Learning rate for the discriminator. Defaults to 2e-4
discriminator_decay (float) | Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6|
batch_size (int) | Number of data samples to process in each step|
discriminator_steps (int) | Number of discriminator updates to do for each generator update. From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper default is 5. Default used is 1 to match original CTGAN implementation|
log_frequency (boolean) | Whether to use log frequency of categorical levels in conditional sampling. Defaults to True|
verbose (boolean) | Whether to have print statements for progress results. Defaults to False|
epochs (int) | Number of training epochs. Defaults to 300|
pac (int) | Number of samples to group together when applying the discriminator. Defaults to 10|
 cuda (bool) | Whether to attempt to use cuda for GPU computation. If this is False or CUDA is not available, CPU will be used. Defaults to True.|

# Citation

If you use **RC-TGAN** for your research, please consider citing the following paper:
<a href="https://arxiv.org/abs/2211.07588">Row Conditional-TGAN for Generating Synthetic Relational Databases</a>
Mohamed Gueye, Yazid Attabi, Maxime Dumas. [Row Conditional-TGAN for Generating Synthetic Relational Databases](https://arxiv.org/abs/2211.07588).

```
@article{gueye2022row,
  title={Row Conditional-TGAN for generating synthetic relational databases},
  author={Gueye, Mohamed and Attabi, Yazid and Dumas, Maxime},
  journal={arXiv preprint arXiv:2211.07588},
  year={2022}
}
```

---


<div align="center">
<a href="https://datacebo.com"><img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/DataCebo.png"></img></a>
</div>
<br/>
<br/>

