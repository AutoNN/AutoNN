from .CNN.cnn_generator import * 
from .CNN.utils.Device import * 
from .CNN.utils.image_augmentation import * 
from .CNN.utils.EDA import *
import pkg_resources as pk 
from .preprocessing.data_cleaning import * 
from .preprocessing import encoding_v3 as enc
from .networkbuilding.final import *
from .exceptions import *


import sys,os 
sys.path.append(os.getcwd())

__all__ = ['exceptions',
            'main',
            'preprocessing',
            'networkbuilding',
            'CNN']

__version__ = pk.get_distribution("nocode-autonn").version 
__authors__ ='Anish Konar, Rajarshi Banerjee, Sagnik Nayak.' 

print(f'''

░█▀▀█ █░░█ ▀▀█▀▀ █▀▀█ ▒█▄░▒█ ▒█▄░▒█ 
▒█▄▄█ █░░█ ░░█░░ █░░█ ▒█▒█▒█ ▒█▒█▒█ 
▒█░▒█ ░▀▀▀ ░░▀░░ ▀▀▀▀ ▒█░░▀█ ▒█░░▀█

Version: {__version__}
An AutoML framework by
Anish Konar, Rajarshi Banerjee, Sagnik Nayak.
''')
