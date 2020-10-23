import os
from config import config
dir = os.path.join(config.baseDir, 'nets')
__all__ = [f[:-3] for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f)) and not f.endswith('__init__.py')]
print(__all__)
