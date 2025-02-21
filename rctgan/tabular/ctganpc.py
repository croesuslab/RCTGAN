"""Wrapper around CTGAN model."""

import numpy as np
from rctgan.ctganpc import PC_CTGANSynthesizer, CTGANSynthesizer
import logging
from rctgan.metadata import Table
from rctgan.tabular.base import BaseTabularModel
import numpy as np
import pandas as pd
import math
import os
import uuid

LOGGER = logging.getLogger(__name__)
COND_IDX = str(uuid.uuid4())
FIXED_RNG_SEED = 73251
TMP_FILE_NAME = '.sample.csv.temp'
DISABLE_TMP_FILE = 'disable'

class CTGANModel(BaseTabularModel):
    """Base class for all the CTGAN models.

    The ``CTGANModel`` class provides a wrapper for all the CTGAN models.
    """

    _MODEL_CLASS = None
    _model_kwargs = None

    _DTYPE_TRANSFORMERS = {
        'O': None
    }

    def _build_model(self):
        return self._MODEL_CLASS(**self._model_kwargs)

    def _fit(self, table_data):
        """Fit the model to the table.

        Args:
            table_data (pandas.DataFrame):
                Data to be learned.
        """
        self._model = self._build_model()

        categoricals = []
        fields_before_transform = self._metadata.get_fields()
        for field in table_data.columns:
            if field in fields_before_transform:
                meta = fields_before_transform[field]
                if meta['type'] == 'categorical':
                    categoricals.append(field)

            else:
                field_data = table_data[field].dropna()
                if set(field_data.unique()) == {0.0, 1.0}:
                    # booleans encoded as float values must be modeled as bool
                    field_data = field_data.astype(bool)

                dtype = field_data.infer_objects().dtype
                try:
                    kind = np.dtype(dtype).kind
                except TypeError:
                    # probably category
                    kind = 'O'
                if kind in ['O', 'b']:
                    categoricals.append(field)
        self._model.fit(
            table_data,
            discrete_columns=categoricals
        )

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates `num_rows` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError(f"{self._MODEL_CLASS} doesn't support conditional sampling.")

    def _set_random_state(self, random_state):
        """Set the random state of the model's random number generator.

        Args:
            random_state (int, tuple[np.random.RandomState, torch.Generator], or None):
                Seed or tuple of random states to use.
        """
        self._model.set_random_state(random_state)


class CTGAN(CTGANModel):
    """Model wrapping ``CTGANSynthesizer`` model.

    Args:
        field_names (list[str]):
            List of names of the fields that need to be modeled
            and included in the generated output data. Any additional
            fields found in the data will be ignored and will not be
            included in the generated output.
            If ``None``, all the fields found in the data are used.
        field_types (dict[str, dict]):
            Dictinary specifying the data types and subtypes
            of the fields that will be modeled. Field types and subtypes
            combinations must be compatible with the SDV Metadata Schema.
        field_transformers (dict[str, str]):
            Dictinary specifying which transformers to use for each field.
            Available transformers are:

                * ``integer``: Uses a ``NumericalTransformer`` of dtype ``int``.
                * ``float``: Uses a ``NumericalTransformer`` of dtype ``float``.
                * ``categorical``: Uses a ``CategoricalTransformer`` without gaussian noise.
                * ``categorical_fuzzy``: Uses a ``CategoricalTransformer`` adding gaussian noise.
                * ``one_hot_encoding``: Uses a ``OneHotEncodingTransformer``.
                * ``label_encoding``: Uses a ``LabelEncodingTransformer``.
                * ``boolean``: Uses a ``BooleanTransformer``.
                * ``datetime``: Uses a ``DatetimeTransformer``.

        anonymize_fields (dict[str, str]):
            Dict specifying which fields to anonymize and what faker
            category they belong to.
        primary_key (str):
            Name of the field which is the primary key of the table.
        constraints (list[Constraint, dict]):
            List of Constraint objects or dicts.
        table_metadata (dict or metadata.Table):
            Table metadata instance or dict representation.
            If given alongside any other metadata-related arguments, an
            exception will be raised.
            If not given at all, it will be built using the other
            arguments or learned from the data.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
        rounding (int, str or None):
            Define rounding scheme for ``NumericalTransformer``. If set to an int, values
            will be rounded to that number of decimal places. If ``None``, values will not
            be rounded. If set to ``'auto'``, the transformer will round to the maximum number
            of decimal places detected in the fitted data. Defaults to ``'auto'``.
        min_value (int, str or None):
            Specify the minimum value the ``NumericalTransformer`` should use. If an integer
            is given, sampled data will be greater than or equal to it. If the string ``'auto'``
            is given, the minimum will be the minimum value seen in the fitted data. If ``None``
            is given, there won't be a minimum. Defaults to ``'auto'``.
        max_value (int, str or None):
            Specify the maximum value the ``NumericalTransformer`` should use. If an integer
            is given, sampled data will be less than or equal to it. If the string ``'auto'``
            is given, the maximum will be the maximum value seen in the fitted data. If ``None``
            is given, there won't be a maximum. Defaults to ``'auto'``.
    """

    _MODEL_CLASS = CTGANSynthesizer

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, plot_loss=False, seed=None,
                 rounding='auto', min_value='auto', max_value='auto'):
        super().__init__(
            field_names=field_names,
            primary_key=primary_key,
            field_types=field_types,
            field_transformers=field_transformers,
            anonymize_fields=anonymize_fields,
            constraints=constraints,
            table_metadata=table_metadata,
            rounding=rounding,
            max_value=max_value,
            min_value=min_value
        )

        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda,
            'plot_loss': plot_loss,
            'seed': seed
        }
        

class PC_CTGANModel(BaseTabularModel):
    """Base class for all the CTGAN models.

    The ``CTGANModel`` class provides a wrapper for all the CTGAN models.
    """

    _MODEL_CLASS = None
    _model_kwargs = None

    _DTYPE_TRANSFORMERS = {
        'O': None
    }
    
    _metadata_pc = None
    
    _type_dict = {}
    _model = None

    def _build_model(self):
        return self._MODEL_CLASS(**self._model_kwargs)

    def _fit(self, table_data, parent_data):
        """Fit the model to the table.

        Args:
            table_data (pandas.DataFrame):
                Data to be learned.
        """
        self._model = self._build_model()

        categoricals = []
        fields_before_transform = self._metadata.get_fields()
        for field in table_data.columns:
            if field in fields_before_transform:
                meta = fields_before_transform[field]
                if meta['type'] == 'categorical':
                    categoricals.append(field)

            else:
                field_data = table_data[field].dropna()
                if set(field_data.unique()) == {0.0, 1.0}:
                    # booleans encoded as float values must be modeled as bool
                    field_data = field_data.astype(bool)

                dtype = field_data.infer_objects().dtype
                try:
                    kind = np.dtype(dtype).kind
                except TypeError:
                    # probably category
                    kind = 'O'
                if kind in ['O', 'b']:
                    categoricals.append(field)
        self._type_dict = dict(table_data.dtypes)
        
        self._model.fit(
            table_data,
            parent_data,
            discrete_columns=categoricals
        )
        
    def fit(self, data, parent_data, field_names_pc=None, field_types_pc=None, field_transformers_pc=None,
                 anonymize_fields_pc=None, primary_key_pc=None, constraints_pc=None, table_metadata_pc=None,
                 rounding_pc='auto', min_value_pc='auto', max_value_pc='auto'):
        if isinstance(data, pd.DataFrame):
            data = data.reset_index(drop=True)

        LOGGER.debug('Fitting %s to table %s; shape: %s', self.__class__.__name__,
                     self._metadata.name, data.shape)
        if not self._metadata_fitted:
            self._metadata.fit(data)

        self._num_rows = len(data)

        LOGGER.debug('Transforming table %s; shape: %s', self._metadata.name, data.shape)
        transformed = self._metadata.transform(data)
        
        
        self._metadata_pc = Table(
                field_names=field_names_pc,
                primary_key=primary_key_pc,
                field_types=field_types_pc,
                field_transformers=field_transformers_pc,
                anonymize_fields=anonymize_fields_pc,
                constraints=constraints_pc,
                dtype_transformers=self._DTYPE_TRANSFORMERS,
                rounding=rounding_pc,
                min_value=min_value_pc,
                max_value=max_value_pc
            )
        self._metadata_pc.fit(parent_data)
        if isinstance(parent_data, pd.DataFrame):
            parent_data = parent_data.reset_index(drop=True)

        LOGGER.debug('Fitting %s to table %s; shape: %s', self.__class__.__name__,
                     self._metadata_pc.name, parent_data.shape)
        self._num_rows_pc = len(parent_data)

        LOGGER.debug('Transforming table %s; shape: %s', self._metadata_pc.name, parent_data.shape)
        transformed_pc = self._metadata_pc.transform(parent_data)

        if self._metadata.get_dtypes(ids=False) and self._metadata_pc.get_dtypes(ids=False):
            LOGGER.debug(
                'Fitting %s model to table %s', self.__class__.__name__, self._metadata.name)
            LOGGER.debug(
                'Fitting %s model to table %s', self.__class__.__name__, self._metadata_pc.name)
            self._fit(transformed, transformed_pc)
    
    def _randomize_samples(self, randomize_samples):
        """Randomize the samples according to user input.

        If ``randomize_samples`` is false, fix the seed that the random number generator
        uses in the underlying models.

        Args:
            randomize_samples (bool):
                Whether or not to randomize the generated samples.
        """
        if self._model is None:
            return

        if randomize_samples:
            self._set_random_state(None)
        else:
            self._set_random_state(FIXED_RNG_SEED)
    
    def _validate_file_path(self, output_file_path):
        """Validate the user-passed output file arg, and create the file."""
        output_path = None
        if output_file_path == DISABLE_TMP_FILE:
            # Temporary way of disabling the output file feature, used by HMA1.
            return output_path

        elif output_file_path:
            output_path = os.path.abspath(output_file_path)
            if os.path.exists(output_path):
                raise AssertionError(f'{output_path} already exists.')

        else:
            if os.path.exists(TMP_FILE_NAME):
                os.remove(TMP_FILE_NAME)

            output_path = TMP_FILE_NAME

        # Create the file.
        with open(output_path, 'w+'):
            pass

        return output_path
    
    def _sample(self, sizes, parent, conditions=None):
        if conditions is None:
            if self._model is None:
                def dupli_rows(data, size_list):
                    df = data.copy()
                    df['index_prim_key'] = range(len(df))
                    size_list_2 = []
                    for k in range(len(size_list)):
                        s = size_list[k]
                        size_list_2 += [k]*s
                    index_prim_key = pd.DataFrame(size_list_2, columns=['index_prim_key'])
                    df = index_prim_key.merge(df, on=['index_prim_key'], how='left')
                    df = df.drop(['index_prim_key'], axis=1)
                    return df
                data = pd.DataFrame(np.zeros((np.sum(sizes), 0)))
                if sum(sizes)==len(sizes):
                    data["Parent_index"] = range(len(data))
                else:
                    data["Parent_index"] = dupli_rows(pd.DataFrame(parent.index, columns=['__index__']), sizes)['__index__']
                return data
            return self._model.sample(sizes, parent)

        raise NotImplementedError(f"{self._MODEL_CLASS} doesn't support conditional sampling.")
    
    
    def sample(self, sizes, parent, randomize_samples=True, output_file_path=None,
               conditions=None):
        if conditions is not None:
            raise TypeError('This method does not support the conditions parameter. '
                            'Please create `sdvrctgan.sampling.Condition` objects and pass them '
                            'into the `sample_conditions` method. '
                            'See User Guide or API for more details.')

        if sizes is None:
            raise ValueError('You must specify the number of rows to sample (e.g. num_rows=100).')

        no_size = True
        for s in sizes:
            if s>0:
                no_size = False
                break
        if no_size or len(sizes)==0:
            return pd.DataFrame()

        self._randomize_samples(randomize_samples)

        output_file_path = self._validate_file_path(output_file_path)
        
        LOGGER.debug('Fitting %s to table %s; shape: %s', self.__class__.__name__,
                     self._metadata_pc.name, parent.shape)
        self._num_rows_pc = len(parent)

        LOGGER.debug('Transforming table %s; shape: %s', self._metadata_pc.name, parent.shape)
        _transformed_pc = self._metadata_pc.transform(parent)

        sampled = self._sample(
            sizes,
            _transformed_pc
        )

        parent_index = sampled["Parent_index"].copy()
        sampled.drop("Parent_index", inplace=True, axis=1)
        sampled = sampled.astype(self._type_dict)
        sampled = self._metadata.reverse_transform(sampled)
        sampled["Parent_index"] = parent_index
        
        return sampled
    
    
    
    def _set_random_state(self, random_state):
        """Set the random state of the model's random number generator.

        Args:
            random_state (int, tuple[np.random.RandomState, torch.Generator], or None):
                Seed or tuple of random states to use.
        """
        self._model.set_random_state(random_state)


class PC_CTGAN(PC_CTGANModel):

    _MODEL_CLASS = PC_CTGANSynthesizer

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, plot_loss=False, seed=None,
                 rounding='auto', min_value='auto', max_value='auto'):
        super().__init__(
            field_names=field_names,
            primary_key=primary_key,
            field_types=field_types,
            field_transformers=field_transformers,
            anonymize_fields=anonymize_fields,
            constraints=constraints,
            table_metadata=table_metadata,
            rounding=rounding,
            max_value=max_value,
            min_value=min_value
        )

        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda,
            'plot_loss': plot_loss,
            'seed': seed
        }
