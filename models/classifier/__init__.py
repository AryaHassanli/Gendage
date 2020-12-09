import os

classifiersDir = 'models/classifier'
__all__ = [f[:-3] for f in os.listdir(classifiersDir) if
           os.path.isfile(os.path.join(classifiersDir, f)) and not f.endswith(
               ('__init__.py', '__pycache__')) and f.endswith('.py')]
