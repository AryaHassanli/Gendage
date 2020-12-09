import os

encodersDir = 'models/encoder'
__all__ = [f[:-3] for f in os.listdir(encodersDir) if
           os.path.isfile(os.path.join(encodersDir, f)) and not f.endswith(
               ('__init__.py', '__pycache__')) and f.endswith('.py')]
