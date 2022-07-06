from tkinter import ttk,messagebox,filedialog as fd
from tkinter import *
from ttkbootstrap import *
import pandas as pd 

class window:
    def __init__(self,root,title='ASC-AutoML Desktop',resolution='640x480'):
        self.root = root
        self.root.title(title)
        self.s = Style()
        self.s.theme_use('cyborg')
        self.root.geometry(resolution)
        self.root.resizable(width=False,height=False)
        
        self.algorithm = ['Random Forest','Decision Tree','Linear Regression',
                        'Logistic Regression','Support Vector Machine','KNN']
        self.combo1 = ttk.Combobox(self.root,values=self.algorithm)
        self.combo1.pack()
        
        self.openfile = ttk.Button(self.root,text='Browse Images or csv files',command=self._selectFile)
        self.openfile.pack()

        # ------------------------TREE VIEW----------------------------
        
        self.tree = ttk.Treeview(self.root,height=23,selectmode='extended')
        self.tree.pack()

    def _selectFile(self):
        # try:
            self.filepath = fd.askopenfilename()
            print(self.filepath)
            if self.filepath.endswith('.csv'):
                df = pd.read_csv(self.filepath)
                self.tree['columns']=df.columns.values
                print(df.describe())
        # except:
        #     pass



if __name__ == '__main__':
    win = Tk()
    window(win)
    win.mainloop()
