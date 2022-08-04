import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('''

░█▀▀█ █░░█ ▀▀█▀▀ █▀▀█ ▒█▄░▒█ ▒█▄░▒█ 
▒█▄▄█ █░░█ ░░█░░ █░░█ ▒█▒█▒█ ▒█▒█▒█ 
▒█░▒█ ░▀▀▀ ░░▀░░ ▀▀▀▀ ▒█░░▀█ ▒█░░▀█

An AutoML framework by
Rajarshi Banerjee, Sagnik Nayak, Anish Konar, Arjun Ghosh.
''')