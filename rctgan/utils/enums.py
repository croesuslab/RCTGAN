from enum import Enum

class FieldType(Enum):
    CATEGORICAL = 'categorical'
    NUMERICAL = 'numerical'
    DATETIME = 'datetime'
    BOOLEAN = 'boolean'
    
class TransformerType(Enum):
    ONE_HOT_ENCODER = 'one_hot_encoder'
    FREQUENCY_ENCODER = 'frequency_encoder'
    GAUSSIAN_NORMALIZER = 'gaussian_normalizer'
    FLOAT_FORMATTER = 'float_formatter'
    TIMESTAMP_ENCODER = 'timestamp_encoder'