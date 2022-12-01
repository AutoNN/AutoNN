from tkinter import *
from tkinter import ttk,messagebox,filedialog
import pandas as pd
from ttkbootstrap import * 
from .CNN.cnn_generator import CreateCNN,CNN
import threading,sys,ctypes,os
from .CNN.utils.EDA import plot_graph
from .CNN.utils.Device import DeviceInfo
from .main import Autonn


timeVar = False
run = True

class TerminalOutput(object):
    def __init__(self,Widget,mode = 'stdout') -> None:
        self.widget = Widget
        self.mode = mode
    
    def write(self,s):
        self.widget.configure(state="normal")
        self.widget.insert("end", s)
        self.widget.configure(state="disabled")
    
    def flush(self):
        pass


class App:
    def __init__(self,root,title,resolution,size) -> None:
        self.root = root
        self.root.title(title)
        self.root.geometry(resolution)
        self.root.resizable(0,0)

        # ---------------- HEADER MENU BAR --------------------------------
        menu = Menu(self.root)
        self.file = Menu(menu)
        self.file.add_command(label='New')
        self.file.add_command(label='Open',command=self.File_open)
        self.file.add_separator()
        self.file.add_command(label='Exit', command=self.root.quit)
        menu.add_cascade(label='File', menu=self.file)
        self.edit = Menu(menu)
        self.edit.add_command(label='Undo')
        menu.add_cascade(label='Edit', menu=self.edit)
        self.root.config(menu=menu)


        # -----------RIGHT CLICK POP UP------------------------------------------------------

        self.men = Menu(self.root, tearoff=False)
        self.men.add_command(label='Clear OUTPUT',command=self.clcOutput)
        self.men.add_command(label='Clear Table',command=self.clcTable)

        self.men.add_separator()
        self.men.add_command(label='Show Graphs',command=self.show_graphs)
        self.men.add_separator()
        self.men.add_command(label='Exit Program', command=self.root.quit)
        self.root.bind('<Button-3>', self.popup)

        # ---------------TABS-
        TABS = ttk.Notebook(self.root)
        TABS.pack()
        csv_frame = Frame(TABS,width=1280,height=300,bg='black')
        image_frame = Frame(TABS,width=1280,height=300,bg='black')
        TABS.add(csv_frame,text='Tabular Dataset')
        TABS.add(image_frame,text= ' Image Dataset')

        F1 = Frame(csv_frame,width=1280,height=20)
        F1.pack()
        F2 = Frame(csv_frame,width=1280,height=280)
        F2.pack()
        # ------------------tabs end-----------------------
        self.nam=StringVar()
        ttk.Label(F1,text='Label Name').grid(row=0,column=0,pady=5,padx=5)
        ttk.Entry(F1,width=20,textvariable=self.nam).grid(row=0,column=1,pady=5,padx=5)

        self.epochs=IntVar()
        ttk.Label(F1,text='Epochs').grid(row=0,column=2,pady=5,padx=5)
        ttk.Entry(F1,width=5,textvariable=self.epochs).grid(row=0,column=3,pady=5,padx=5)

        # BUTTONS
        ttk.Button(F1,text = 'Open File',width=20,style='info.TButton',
        command=self.File_open).grid(row=0,column=4,pady=5,padx=5)

        ttk.Button(F1,text = 'Start Training',width=20,style='success.TButton',
        command=self.start_training_csv).grid(row=0,column=6,pady=5,padx=5)
        ttk.Button(F1,text = 'Save the Model',width=20,style='success.Outline.TButton',
        command=self.SaveCsvModel).grid(row=0,column=7,pady=5,padx=5)

        # ttk.Label(F1,text='Progress').grid(row=1,column=0,pady=5,padx=5)
        # self.pb1 = ttk.Progressbar(F1, value=0,length=750,
        #  style='success.Horizontal.TProgressbar')
        # self.pb1.grid(row=1,column=1,columnspan=5)

        # RADIO BUTTON------------
        self.split = BooleanVar()
        ttk.Radiobutton(image_frame,text='Split required',variable=self.split,value=True,width=20,
        style='danger.Outline.Toolbutton').grid(row=1,column=6)
        ttk.Radiobutton(image_frame,text='Split NOT required',variable=self.split,value=False,width=20,
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


        self.imgEpoch=IntVar()
        ttk.Label(image_frame,text='Epochs').grid(row=0,column=2,pady=5,padx=5)
        ttk.Entry(image_frame,width=10,textvariable=self.imgEpoch).grid(row=0,column=3,pady=5,padx=5)

        self.lr= DoubleVar()
        ttk.Label(image_frame,text='Learning Rate').grid(row=0,column=0,pady=5,padx=5)
        ttk.Entry(image_frame,width=10,textvariable=self.lr).grid(row=0,column=1,pady=5,padx=5)
        self.lr.set(0.003)
        # BUTTONS
        ttk.Button(image_frame,text = 'Open folder',width=20,style='info.TButton',
        command=self.get_img_dataset).grid(row=0,column=4,pady=5,padx=5)
        ttk.Button(image_frame,text = 'Start Training',width=20,style='success.Outline.TButton',
        command=self.Start_training).grid(row=0,column=5,pady=5,padx=5)
        ttk.Button(image_frame,text = 'Show Configs',width=20,
        command=self.show_all_configurations).grid(row=0,column=6,pady=5,padx=5)
        self.savecnn_btn=ttk.Button(image_frame,text = 'Save Trained Model',width=20,style='warning.Outline.TButton',
        command=self.save_model)
        self.savecnn_btn.grid(row=1,column=5,pady=5,padx=5)
        self.savecnn_btn['state']='disabled'

        # ttk.Label(image_frame,text='Progress').grid(row=1,column=0,pady=5,padx=5)
        self.pb2 = ttk.Progressbar(image_frame, value=0,length=1280,mode='indeterminate',
         style='success.Horizontal.TProgressbar')
        
        self.pb1 = ttk.Progressbar(F1, value=0,length=1280,mode='indeterminate',
         style='success.Horizontal.TProgressbar')

        
        self.channels = IntVar()
        self.numclass = IntVar()
        ttk.Entry(image_frame,width=10,textvariable=self.channels).grid(row=1,column=1)
        ttk.Entry(image_frame,width=10,textvariable=self.numclass).grid(row=1,column=2)
        self.channels.set('#channels')
        self.numclass.set('#Classes')

        self.display_btn = ttk.Button(image_frame,text='Display Graphs',command=self.show_graphs)
        self.display_btn.grid(row=1,column=3)
        self.display_btn['state']='disabled'
                # for loading any trained cnn model 
        self.load_cnn = ttk.Button(image_frame,text='Load Model',command=self.load_cnn_model,width=20)
        self.load_cnn.grid(row=1,column=4)

        self.disp = Text(image_frame,height=10,width=130,background='black',foreground='lime')
        self.disp.grid(row=4,column=0,columnspan=7)

        self.disp1 = Text(image_frame,height=10,width=40,background='black',foreground='yellow')
        self.disp1.grid(row=4,column=7,columnspan=4)


        # ----combobox------------
        self.batch_sizes = ttk.Combobox(image_frame,values=[2**i for i in range(9)])
        self.batch_sizes.grid(row=0,column=7)
        self.batch_sizes.set('Select batch size')
        self.batch_sizes['state']='readonly'

        ttk.Label(self.root,text='OUTPUT').pack()
        self.textBox = Text(self.root,height=17,width=180)
        self.textBox.pack()

        self.clockwid = ttk.Label(image_frame)
        self.clockwid.grid(row=1,column=0)
        info = DeviceInfo()
        x_ =ttk.Label(self.root,text='', font=("Arial", 9),foreground='yellow')
        x_.pack(side='right')
        ttk.Label(self.root,text= u"  \u00A9"+ 'AutoNN', font=("Arial", 9)).pack(side='left')
        sys.stdout = TerminalOutput(self.textBox)
        usage = threading.Thread(target=App._usages,args=(info,self.disp1,x_))
        usage.start()
        root.protocol("WM_DELETE_WINDOW", self.on_closing)


    

    @staticmethod
    def _usages(obj,widget,w2):
        global run 
        w2.config(text=obj.getusage)
        widget.configure(state='normal')
        widget.delete('1.0',END)
        widget.insert('end',obj.getDeviceInfo)
        widget.configure(state='disabled')
        if run:
            widget.after(1000,lambda :App._usages(obj,widget,w2))



    @staticmethod
    def Timer(widget,clock):
        global timeVar

        if timeVar:
            clock +=1
            widget.config(text='{:.2f} mins'.format(clock/60))
            widget.after(1000,lambda :App.Timer(widget,clock))


    def load_cnn_model(self):
        
        path = filedialog.askopenfilename(initialdir='./best_models/',
        title='Select Trained Model file',
        filetypes=(('Model files','*.pth'),('model files','*.pt'),
        ('All files','*.*'))
        )
        configfile = (path.split('/')[-1]).split('.')[0] + '.json'
        try:
            self.new_model = CNN(self.channels.get(),self.numclass.get())
            self.new_model.load(PATH=path,
            config_path=f'./config_files/{configfile}',
            printmodel=True)
            self.new_model.summary((3,32,32))
        except:
            messagebox.showerror('Invalid Input','Make sure #channels and #classes\n are INTEGER VALUES.')
    # -------Progress bar FOR IMAGE TRAINING________________

    def get_img_dataset(self):
        self.folder = filedialog.askdirectory(title='Open Folder')
        self.gen_cnn_object = CreateCNN()

    def show_all_configurations(self):
        try:
            self.disp.configure(state="normal")
            self.disp.delete('1.0',END)
            self.disp.insert(END,f'''
            Training Set Path:  {self.folder}
            Epochs:             {self.imgEpoch.get()}
            Split Required:     {self.split.get()}
            Batch Size:         {self.batch_sizes.get()}
            Learning Rate:      {self.lr.get()}
                ''')
            self.disp.configure(state="disabled")

        except Exception as e:
            messagebox.showerror('Empty Path','Please provide the training folder path.\n click "Open Folder"')
    
    
    def __Func(self):
        global timeVar
        timeVar=True
        self.pb2.grid(row=3,column=0,columnspan=20)
        self.pb2.start()
        self.cnn_model,bestconfig,self.history =self.gen_cnn_object.get_bestCNN(
        path_trainset=self.folder,
        split_required=self.split.get(), 
        batch_size=int(self.batch_sizes.get()), 
        EPOCHS= self.imgEpoch.get(),
        LR=self.lr.get()
        )
        self.display_btn['state']='normal'
        self.savecnn_btn['state']='normal'
        print('Trainig Completed!')
        self.textBox.insert('end',self.cnn_model.__str__())
        print('History List: ',self.history)
        print('Best Configuration architecture: ',bestconfig)
        self.pb2.stop()
        self.pb2.grid_remove()
        timeVar = False

    def Start_training(self):
        
        clock_thread = threading.Thread(target=App.Timer,args=(self.clockwid,0))
        process1 = threading.Thread(target=self.__Func)
        process1.start()
        clock_thread.start()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            sys.exit()

    def save_model(self):
        def save(x):
            self.cnn_model.save(filename=f'{x}.pth',config_filename=f'{x}.json')
            messagebox.showinfo('Model Saved',
            f'Model saved at location "./best_models/{x}.pth"')
            pop_model_save_window.destroy()

        
        pop_model_save_window = Toplevel(self.root)
        pop_model_save_window.geometry('300x100')
        name = StringVar()
        ttk.Label(pop_model_save_window, text = 'Save Model as').pack()
        ttk.Entry(pop_model_save_window,textvariable=name).pack()
        ttk.Button(pop_model_save_window,text="SAVE",width=18,
        command = lambda :save(name.get())).pack()
        

    # -----------------methods to control csv datasets------------------

    def __start_training_csv(self,a,b):
        atonn = Autonn(a,b)
        atonn.preprocessing()
        atonn.neuralnetworkgeneration()
        self.pb1.stop()
        pass

    def start_training_csv(self):
        self.pb1.start()
        self.pb1.grid(row=3,column=0,columnspan=20)
        p1 = threading.Thread(target=self.__start_training_csv,args=(self.csv_file,self.nam.get()))
        p1.start()

    def SaveCsvModel(self):
        pass

    def File_open(self):
        self.csv_file = filedialog.askopenfilename(title='Open CSV file',
        filetypes=(('csv files','*.csv'),('xlsx files','*.xlxs'),
        ('All files','*.*')))
        df = None
        try:
            if self.csv_file.endswith('.csv'):
                df = pd.read_csv(self.csv_file)
            elif self.csv_file.endswith('.xlxs'):
                df = pd.read_excel(self.csv_file)
            else:
                pass
            
        except Exception:   
                messagebox.showerror('ERROR!','Invalid File! Unable to open file!') 
        if self.csv_file:
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

            # -------OTHER FUNCTIONALITIES--------       

    def clcOutput(self):
        '''
        To Clear the OUTPUT display
        '''
        self.textBox.configure(state="normal")
        self.textBox.delete('1.0',END)
        self.textBox.configure(state="disabled")
        
    def clcTable(self):
        for child in self.tree.get_children():
            self.tree.delete(child)
         

    def show_graphs(self):
        plot_graph(self.history)
        

    def popup(self,e):
        self.men.tk_popup(e.x_root,e.y_root)


def main():
    if os.name =='nt':
        ctypes.windll.shcore.SetProcessDpiAwareness(0)
    win = Style(theme='darkly').master 
    App(win,'AutoNN GUI','1280x720',40)
    win.mainloop()

if __name__=='__main__':
    main()