import sys, types, importlib.abc, importlib.machinery

# Stub for pyonmttok
if 'pyonmttok' not in sys.modules:
    import pyonmttok
    sys.modules['pyonmttok'] = pyonmttok

# Stub for fasttext
if 'fasttext' not in sys.modules:
    import fasttext
    sys.modules['fasttext'] = fasttext

# Stub for rapidfuzz
if 'rapidfuzz' not in sys.modules:
    import rapidfuzz
    sys.modules['rapidfuzz'] = rapidfuzz

# Fallback dynamic class generator for pickle unpickling legacy checkpoints
class _DummyClass:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state):
        if isinstance(state, dict): self.__dict__.update(state)
        elif isinstance(state, (list, tuple)): self.data = list(state)
    def __repr__(self): return f"<{self.__class__.__name__}>"

class _FallbackLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        def _get(name):
            if name not in mod.__dict__:
                cls = type(name, (_DummyClass,), {'__module__': spec.name})
                setattr(mod, name, cls)
            return getattr(mod, name)
        mod.__getattr__ = _get
        return mod
    def exec_module(self, module): pass

class _FallbackFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('onmt.') or fullname.startswith('torchtext'):
            return importlib.machinery.ModuleSpec(fullname, _FallbackLoader())
        return None

if not any(isinstance(f, _FallbackFinder) for f in sys.meta_path):
    sys.meta_path.append(_FallbackFinder())

# Ensure torch.load defaults to weights_only=False for OpenNMT checkpoints under PyTorch 2.6+
try:
    import torch
    import torch.serialization
    if not getattr(torch.load, '_safe_patched', False):
        _real_load = torch.serialization.load
        def _safe_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return _real_load(*args, **kwargs)
        _safe_load._safe_patched = True
        torch.load = _safe_load
        torch.serialization.load = _safe_load
except ImportError:
    pass

