import json
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class Config:
    sdtypes: Dict[str, str] = field(default_factory=dict)
    transformers: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        """Pretty print the dictionary."""
        config = {
            'sdtypes': self.sdtypes,
            'transformers': {k: repr(v) for k, v in self.transformers.items()}
        }

        printed = json.dumps(config, indent=4)
        for transformer in self.transformers.values():
            quoted_transformer = f'"{transformer}"'
            if quoted_transformer in printed:
                printed = printed.replace(quoted_transformer, repr(transformer))

        return printed