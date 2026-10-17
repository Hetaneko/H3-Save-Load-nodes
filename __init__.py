from .nodes import (
    NODE_CLASS_MAPPINGS as _condition_nodes,
    NODE_DISPLAY_NAME_MAPPINGS as _condition_names,
)
from .latent_nodes import (
    NODE_CLASS_MAPPINGS as _latent_nodes,
    NODE_DISPLAY_NAME_MAPPINGS as _latent_names,
)

NODE_CLASS_MAPPINGS = {**_condition_nodes, **_latent_nodes}
NODE_DISPLAY_NAME_MAPPINGS = {**_condition_names, **_latent_names}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
