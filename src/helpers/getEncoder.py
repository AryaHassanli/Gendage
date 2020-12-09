import sys
# noinspection PyUnresolvedReferences
from models.encoder import *


# TODO: move getEncoder to models/encoder __init__

def get(encoder_name, **kwargs):
    encoder = getattr(eval(encoder_name), encoder_name)
    if callable(encoder):
        print(encoder_name + " Found!")
        encoder = encoder(**kwargs)
    else:
        sys.exit('encoder not found')

    return encoder
