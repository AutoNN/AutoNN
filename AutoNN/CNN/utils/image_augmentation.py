from PIL import  Image
import os
from tqdm import  tqdm


class Augment(object):
    def __init__(self,path) -> None:
        self.path = path

    def augment(self):
        for folder in os.listdir(self.path):
            print(f'Current folder: {folder}')
            for image in tqdm(os.listdir(self.path+f'/{folder}')):
                if image.lower().endswith(('.jpg','.png','.jpeg','.tif','.tiff')):
                    x = Image.open(self.path+f'/{folder}/{image}')
                    x45 = x.rotate(45)
                    x136 = x.rotate(136)
                    x279 = x.rotate(279)
                    x330 = x.rotate(330)
                    x45.save(f'{self.path}/{folder}/r45{image}')
                    x136.save(f'{self.path}/{folder}/r136{image}')
                    x279.save(f'{self.path}/{folder}/r279{image}')
                    x330.save(f'{self.path}/{folder}/r330{image}')
                    hflip = x.transpose(Image.FLIP_LEFT_RIGHT)
                    vflip = x.transpose(Image.FLIP_TOP_BOTTOM)
                    hflip.save(f'{self.path}/{folder}/h{image}')
                    vflip.save(f'{self.path}/{folder}/v{image}')
                else:
                    raise("Make sure you have the correct image format")