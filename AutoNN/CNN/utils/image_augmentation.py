from PIL import  Image
import os
from tqdm import  tqdm
from AutoNN.exceptions import InvalidImageFileError,InvalidFolderStructureError


class Augment(object):
    def __init__(self,path) -> None:
        '''
        Args: 
            path: provide the path to your image folder
            which you want to augment

            ../Folder/dataset/cats/x1.png
            ../Folder/dataset/cats/x2.png
            .
            .
            .
            ../Folder/dataset/dogs/xx1.png
            ../Folder/dataset/dogs/xx2.png
            ../Folder/dataset/dogs/xx3.png
            .
            .
            path = '../Folder/dataset/'
        '''
        self.path = path
        self.dataset = dict()
        self.dist = dict()
        try:
            for class_folder in os.listdir(path):
                x =len(os.listdir(os.path.join(path,class_folder)))
                self.dataset[os.path.join(path,class_folder)]=x
                self.dist[class_folder]=x
            
            idex = max(self.dataset.keys(),key = lambda k:self.dataset[k])
            self.dset={k:[self.dataset[idex]-v,self.dataset[idex]/v] for k,v in self.dataset.items()}
        except:
            raise InvalidFolderStructureError
            
    
    def get_info(self):
        temp = dict()
        for class_folder in os.listdir(self.path):
            temp[class_folder]=len(os.listdir(os.path.join(self.path,class_folder)))
        print("Before Augmentation Distribution: ")
        print(self.dist)
        print("After Augmentation Distribution: ")
        print(temp)

    def __augment(self,img_path,image,n):
        a = [45,136,279,330,Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM]
        for i in range(n):
            try:
                x =Image.open(os.path.join(img_path,image))
                if i<4:
                    x_ = x.rotate(a[i])
                    x_.save(os.path.join(img_path,f'r{a[i]}{image}'))
                else:
                    x_=x_.transpose(a[i])
                    x_.save(os.path.join(img_path,f'flip{i}{image}'))
            except :
                raise InvalidImageFileError
                

    def augment(self):
        for folder_path,n in self.dset.items():
            print(f'Current folder: {os.path.split(folder_path)[-1]}')
            if n[-1]>1.7:
                for image in tqdm(os.listdir(folder_path)):
                    self.__augment(folder_path,image,int(n[-1])-1)

        print('Dataset augmentation Complete!!')

