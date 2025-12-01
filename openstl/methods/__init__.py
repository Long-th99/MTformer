from .MTFormer import MTformer

method_maps = {
    'mtformer': MTformer,
}

__all__ = [
    'method_maps', 'MTformer'
]