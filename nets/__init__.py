import os

from config import config

netsDir = os.path.join(config.baseDir, 'nets')
__all__ = [f[:-3] for f in os.listdir(netsDir) if
           os.path.isfile(os.path.join(netsDir, f)) and not f.endswith('__init__.py')]
