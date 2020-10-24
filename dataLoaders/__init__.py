import os

from config import config

dataLoadersDir = os.path.join(config.baseDir, 'dataLoaders')
__all__ = [f[:-3] for f in os.listdir(dataLoadersDir) if
           os.path.isfile(os.path.join(dataLoadersDir, f)) and not f.endswith('__init__.py')]
