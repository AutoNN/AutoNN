from torch import device,cuda
import psutil

class DeviceInfo:
    def __init__(self):
        self.device = device('cuda' if cuda.is_available() else 'cpu')

    @property
    def getDeviceInfo(self):
        str = f'Using Device: {self.device}'    
        if self.device.type =='cuda':
            str+= f"\nDevice Name: {cuda.get_device_name()}\nMemory Usage:\n"
            str+= f'Allocated: {(cuda.memory_allocated()*1e-6):.2f} MB'
            str+= f'\nCached: {(cuda.memory_reserved()*1e-6):.2f} MB\n'
        return str

    @property
    def getusage(self,bars=20):
        str=''
        x,y = psutil.cpu_percent(),psutil.virtual_memory().percent
        CPU = (x/100.0)
        bar1 = '█'*int(CPU*bars)+' '*(bars-int(CPU*bars))
        MEM = (y/100.0)
        bar2 = '█'*int(MEM*bars)+' '*(bars-int(MEM*bars))

        str+=f"\rCPU usage: |{bar1}| {x:.2f}%   "
        str+=f"\tMEMORY Usage: |{bar2}| {y:.2f}%  \r"
        return str