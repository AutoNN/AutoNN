from tkinter import *
from tkinter import ttk
import os
from csv import DictReader
from ttkbootstrap import * 
from PIL import ImageTk, Image


class window:
    def __init__(self,root,title,resolution,size) -> None:
        self.root = root
        self.root.title(title)
        self.root.geometry(resolution)
        self.root.resizable(0,0)

        # load_logo = Image.open(os.path.join(os.getcwd(),'AutoNN','assets','1logo.png'))
        # load_logo = load_logo.resize((1280, 150), Image.ANTIALIAS)

        # render_image = ImageTk.PhotoImage(load_logo)
        # img = Label(self.root,image=render_image)
        # img.image =  render_image
        # img.grid(row=0,column=0,columnspan=15)
        Label(self.root, text='AutoNN', font='Ariel ' + '50',bg="#000000",fg="#005A9C", bd=5,pady=5,padx=520
              ).grid(row=0, column=1, columnspan=14, pady=0, padx=0)

        # ---------------- HEADER MENU BAR --------------------------------
        menu = Menu(self.root)
        self.file = Menu(menu)
        self.file.add_command(label='New')
        self.file.add_command(label='Open')
        self.file.add_command(label='Save',command=self.saveModel)
        self.file.add_separator()
        self.file.add_command(label='Exit', command=self.root.quit)
        menu.add_cascade(label='File', menu=self.file)
        self.edit = Menu(menu)
        self.edit.add_command(label='Undo')
        menu.add_cascade(label='Edit', menu=self.edit)
        self.root.config(menu=menu)
        # -----------RIGHT CLICK POP UP------------------------------------------------------

        self.men = Menu(self.root, tearoff=False)
        self.men.add_command(label='Clear Table',command=self.clcTable)
        self.men.add_separator()
        self.men.add_command(label='Show All',command=self.showALL)
        self.men.add_separator()
        self.men.add_command(label='Exit Database', command=self.root.quit)
        self.root.bind('<Button-3>', self.popup)

        # ---------------TABS-
        TABS = ttk.Notebook(self.root)
        TABS.grid(rows=1,column=0,columnspan=10,rowspan=10)
        csv_frame = Frame(TABS,width=1000,height=300,bg='black')
        image_frame = Frame(TABS,width=1000,height=300,bg='blue')
        TABS.add(csv_frame,text='Tabular Dataset')
        TABS.add(image_frame,text= ' Image Dataset')

        F1 = Frame(csv_frame,width=1000,height=20)
        F1.pack()
        F2 = Frame(csv_frame,width=1000,height=280)
        F2.pack()

        self.nam=StringVar()
        ttk.Label(F1,text='Path to Dataset').pack(side=LEFT,pady=5,padx=5)
        ttk.Entry(F1,width=40,textvariable=self.nam).pack(side=LEFT,pady=5,padx=5)

        self.nam=IntVar()
        ttk.Label(F1,text='Epochs').pack(side=LEFT,pady=5,padx=5)
        ttk.Entry(F1,width=5,textvariable=self.nam).pack(side=LEFT,pady=5,padx=5)

        # BUTTONS
        ttk.Button(F1,text = 'View Csv',width=20,
        command=self.View_contents).pack(side=LEFT,pady=5,padx=5)
        ttk.Button(F1,text = 'Start Training',width=20,
        command=self.start_training_csv).pack(side=RIGHT,pady=5,padx=5)


        # -----------Tree view-----
        scrollbarx = Scrollbar(F2, orient=HORIZONTAL)
        scrollbary = Scrollbar(F2, orient=VERTICAL)
        self.tree = ttk.Treeview(F2,height=15,selectmode='extended',
         yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=self.tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=self.tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        self.tree.pack(fill=BOTH, expand=1)

        

        # ---TERMINAL-------------

        # ----------------

    def start_training_csv(self,path):
        pass 



    def View_contents(self,path):
        '''
        Args: 
            path: path to csv file
        '''
        with open(path) as f:
            reader = DictReader(f,delimiter=',')
            for row in reader:
                print(row)
        


    def saveModel(self):
        pass 

    def clcTable(self):
        pass 

    def showALL(self):
        pass 

    def popup(self,e):
        self.men.tk_popup(e.x_root,e.y_root)


if __name__ == '__main__':
    win = Style(theme='darkly').master 
    window(win,'AutoNN GUI','1280x720',40)
    win.mainloop()
