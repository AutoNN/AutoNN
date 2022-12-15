import pkg_resources as pk 


import sys,os 
sys.path.append(os.path.dirname(__file__))


__all__ = ['CNN','networking','preprocessing']

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
