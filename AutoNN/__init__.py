from .CNN import cnn_generator
from .CNN.utils import Device 
from .CNN.utils import image_augmentation 
from .CNN.utils import EDA
import pkg_resources as pk 
from .preprocessing.data_cleaning import * 
from .preprocessing import encoding_v3 as enc
from .networkbuilding.final import *
from .exceptions import *




__all__ = ['cnn_generator',
'Device',
'image_augmentation',
'EDA']

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
