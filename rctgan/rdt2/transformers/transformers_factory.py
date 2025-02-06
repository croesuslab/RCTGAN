from rctgan.utils.enums import FieldType, TransformerType
from rctgan.rdt2.transformers.numerical import FloatFormatter, GaussianNormalizer
from rctgan.rdt2.transformers.categorical import FrequencyEncoder, OneHotEncoder
from rctgan.rdt2.transformers.datetime import OptimizedTimestampEncoder

class TransformerFactory:
    @staticmethod
    def get_transformer(field_type, transformer_type=None, format_datetime=None):
        if field_type == FieldType.CATEGORICAL:
            if transformer_type == TransformerType.ONE_HOT_ENCODER:
                return OneHotEncoder()
            else:
                return FrequencyEncoder(add_noise=True)
        elif field_type == FieldType.NUMERICAL:
            if transformer_type == TransformerType.GAUSSIAN_NORMALIZER:
                return GaussianNormalizer()
            elif transformer_type == TransformerType.FLOAT_FORMATTER:
                return FloatFormatter(missing_value_replacement='mean')
            else:
                return GaussianNormalizer()
        elif field_type == FieldType.DATETIME:
            return OptimizedTimestampEncoder(missing_value_replacement='mean', datetime_format=format_datetime)
        else:
            raise ValueError(f"Unsupported field type: {field_type}")