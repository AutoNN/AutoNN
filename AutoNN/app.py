from tkinter import *
from tkinter import ttk,messagebox,filedialog
import os
import pandas as pd
import numpy as np
from ttkbootstrap import * 
# from PIL import ImageTk, Image
from CNN.cnn_generator import CreateCNN

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
        self.file.add_command(label='Open',command=self.File_open)
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
        self.men.add_command(label='Show All Configurations',command=self.show_all_configurations)
        self.men.add_separator()
        self.men.add_command(label='Exit Database', command=self.root.quit)
        self.root.bind('<Button-3>', self.popup)

        # ---------------TABS-
        TABS = ttk.Notebook(self.root)
        TABS.grid(rows=1,column=0,columnspan=10,rowspan=10)
        csv_frame = Frame(TABS,width=1000,height=300,bg='black')
        image_frame = Frame(TABS,width=1000,height=300,bg='black')
        TABS.add(csv_frame,text='Tabular Dataset')
        TABS.add(image_frame,text= ' Image Dataset')

        F1 = Frame(csv_frame,width=1280,height=20)
        F1.pack()
        F2 = Frame(csv_frame,width=1280,height=280)
        F2.pack()

        self.nam=StringVar()
        ttk.Label(F1,text='Path to Dataset').grid(row=0,column=0,pady=5,padx=5)
        ttk.Entry(F1,width=40,textvariable=self.nam).grid(row=0,column=1,pady=5,padx=5)

        self.epochs=IntVar()
        ttk.Label(F1,text='Epochs').grid(row=0,column=2,pady=5,padx=5)
        ttk.Entry(F1,width=5,textvariable=self.epochs).grid(row=0,column=3,pady=5,padx=5)

        # BUTTONS
        ttk.Button(F1,text = 'Open File',width=20,style='info.TButton',
        command=self.File_open).grid(row=0,column=4,pady=5,padx=5)
        # ttk.Button(F1,text = 'View Csv',width=20,style='info.TButton',
        # command=self.View_contents).grid(row=0,column=5,pady=5,padx=5)
        ttk.Button(F1,text = 'Start Training',width=20,style='success.TButton',
        command=self.start_training_csv).grid(row=0,column=6,pady=5,padx=5)
        ttk.Button(F1,text = 'Save the Model',width=20,style='success.Outline.TButton',
        command=self.SaveCsvModel).grid(row=0,column=7,pady=5,padx=5)

        ttk.Label(F1,text='Progress').grid(row=1,column=0,pady=5,padx=5)
        ttk.Progressbar(F1, value=0,length=750,
         style='success.Horizontal.TProgressbar').grid(row=1,column=1,columnspan=5)

        # RADIO BUTTON------------
        self.split = BooleanVar()
        ttk.Radiobutton(image_frame,text='Split required',variable=self.split,value=True,
        style='danger.Outline.Toolbutton').grid(row=1,column=6)
        ttk.Radiobutton(image_frame,text='Split NOT required',variable=self.split,value=False,
        style='danger.Outline.Toolbutton').grid(row=1,column=7)




        # -----------Tree view-----
        self.tree = ttk.Treeview(F2,height=15,selectmode='extended',show='headings')
        scrollbarx = Scrollbar(F2, orient='horizontal',command=self.tree.xview)
        scrollbary = Scrollbar(F2, orient='vertical',command=self.tree.yview)
        
        
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.pack(side=BOTTOM, fill=X)
        self.tree.pack(fill=BOTH, expand=1)
        self.tree.config(xscrollcommand=scrollbarx.set)
        self.tree.config(yscrollcommand=scrollbary.set)

        self.tree.column('#0',width=0,stretch=NO)
        self.tree.heading('#0',text=None)

        F2.pack_propagate(0)


        # ---------------IMAGE FRAME----FOR IMAGE DATASET------------------------------


        self.imgPath=StringVar()
        ttk.Label(image_frame,text='Path to Dataset').grid(row=0,column=0,pady=5,padx=5)
        ttk.Entry(image_frame,width=40,textvariable=self.imgPath).grid(row=0,column=1,pady=5,padx=5)

        self.imgEpoch=IntVar()
        ttk.Label(image_frame,text='Epochs').grid(row=0,column=2,pady=5,padx=5)
        ttk.Entry(image_frame,width=5,textvariable=self.imgEpoch).grid(row=0,column=3,pady=5,padx=5)

        self.lr= DoubleVar()
        ttk.Label(image_frame,text='Learning Rate').grid(row=1,column=3,pady=5,padx=5)
        ttk.Entry(image_frame,width=20,textvariable=self.lr).grid(row=1,column=4,pady=5,padx=5)
        self.lr.set(0.003)
        # BUTTONS
        ttk.Button(image_frame,text = 'Open folder',width=20,style='info.TButton',
        command=self.get_img_dataset).grid(row=0,column=4,pady=5,padx=5)
        ttk.Button(image_frame,text = 'Start Training',width=20,style='success.TButton',
        command=self.Start_training).grid(row=0,column=5,pady=5,padx=5)
        # ttk.Button(image_frame,text = 'Save the Model',width=20,
        # command=self.).grid(row=0,column=6,pady=5,padx=5)
        ttk.Button(image_frame,text = 'Show Configs',width=20,
        command=self.show_all_configurations).grid(row=0,column=6,pady=5,padx=5)

        ttk.Label(image_frame,text='Progress').grid(row=1,column=0,pady=5,padx=5)
        ttk.Progressbar(image_frame, value=0,length=350,mode='determinate',
         style='success.Horizontal.TProgressbar').grid(row=1,column=1,columnspan=2)

        self.disp = ttk.Label(image_frame)
        self.disp.grid(row=3,column=0,columnspan=20)

        # ----combobox------------
        self.batch_sizes = ttk.Combobox(image_frame,values=[2**i for i in range(9)])
        self.batch_sizes.grid(row=0,column=7)
        self.batch_sizes.set('Select batch size')
        self.batch_sizes['state']='readonly'

        # ---TERMINAL-------------

    def get_img_dataset(self):
        self.folder = filedialog.askdirectory(title='Open Folder')
        self.gen_cnn_object = CreateCNN()

    def show_all_configurations(self):
        try:
            self.disp.config(text=f'Training Set Path: {self.folder}\nEpochs: {self.imgEpoch.get()}\nSplit Required: {self.split.get()}\nBatch Size: {self.batch_sizes.get()}')
        except:
            pass 


    def Start_training(self):
        self.cnn_model,self.cnn_bestconfig,self.cnn_history=self.gen_cnn_object.get_bestCNN(path_trainset=self.folder,
        split_required=self.split.get(), 
        batch_size=int(self.batch_sizes.get()), 
        EPOCHS = self.imgEpoch.get(),
        LR=self.lr.get(),
        )
    
    def save_model(self):
        def save(x):
            self.cnn_model.save(filename=f'{x}.pth')
            messagebox.showinfo('Model Saved',
            f'Model saved at location "./best_models/{x}.pth"')
            pass

        
        pop_model_save_window = Toplevel(self.root)
        pop_model_save_window.geometry('300x100')
        name = StringVar()
        ttk.Label(pop_model_save_window, text = 'Save Model as').pack()
        ttk.Entry(pop_model_save_window,textvariable=name).pack()
        ttk.Button(pop_model_save_window,text="SAVE",width=18,
        command = lambda :save(name.get())).pack()
        


    # -----------------methods to control csv datasets------------------


    def start_training_csv(self):
        print(self.split.get())
        pass 

    def SaveCsvModel(self):
        pass

    def View_contents(self):
       

        pass 

    def File_open(self):
        file = filedialog.askopenfilename(title='Open CSV file',
        filetypes=(('csv files','*.csv'),('xlsx files','*.xlxs'),
        ('All files','*.*')))
        df = None
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.endswith('.xlxs'):
                df = pd.read_excel(file)
            else:
                pass
            
        except Exception:   
                messagebox.showerror('ERROR!','Invalid File! Unable to open file!') 
        
        self.tree['column']=list(df.columns)
        self.tree['show']='headings'
        for column in self.tree['column']:
        # for i,column in enumerate(df.columns):
            self.tree.column(column,width=90,minwidth=100,stretch=False)
            self.tree.heading(column,text=column)
        self.tree.update()    
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            self.tree.insert('','end',values=row)
    
    def saveModel(self):
        pass 

    def clcTable(self):
        pass 

    def showALL(self):
        pass 

    def popup(self,e):
        self.men.tk_popup(e.x_root,e.y_root)


win = Style(theme='darkly').master 
window(win,'AutoNN GUI','1280x720',40)
win.mainloop()
