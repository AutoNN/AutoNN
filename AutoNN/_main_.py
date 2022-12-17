import threading,sys,ctypes,os,json
from tkinter import *
from tkinter import ttk,messagebox,filedialog
import pandas as pd
from ttkbootstrap import * 
import webbrowser
from AutoNN.CNN.cnn_generator import CreateCNN,CNN
from AutoNN.CNN.utils.EDA import plot_graph
from AutoNN.CNN.utils.Device import DeviceInfo
from AutoNN.CNN.models.resnet import resnet
from AutoNN.CNN.utils.image_augmentation import Augment
from AutoNN.main import Autonn

timeVar = False
run = True
switch = True

PATH2JSON = os.path.join(os.path.dirname(__file__),'default_config.json')


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
    def __init__(self,root,title,resolution) -> None:
        self.root = root
        self.root.title(title)
        self.root.geometry(resolution)
        self.root.resizable(0,0)

        # ---------------- HEADER MENU BAR --------------------------------
        menu = Menu(self.root)
        self.file = Menu(menu)
        self.file.add_command(label='Open CSV File',command=self.File_open)
        self.file.add_command(label='Clear OUTPUT',command=self.clcOutput)
        self.file.add_command(label='Clear Table',command=self.clcTable)
        self.file.add_separator()
        self.file.add_command(label='Exit', command=self.on_closing)
        menu.add_cascade(label='File', menu=self.file)
        edit = Menu(menu)
        edit.add_command(label='Path Settings',command=self.__path_settings)
        menu.add_cascade(label='Edit', menu=edit)
        self.HELP = Menu(menu)
        self.HELP.add_command(label='Help T_T',command= lambda : webbrowser.open("https://autonn.github.io/AutoNN/gui/lesson2/"))
        self.HELP.add_command(label='About ', command = lambda: webbrowser.open("https://autonn.github.io/AutoNN/about/"))
        self.HELP.add_separator()
        self.HELP.add_command(label="DO NOT Click ME!!", command = lambda: webbrowser.open("https://www.youtube.com/shorts/aJ2POGrAp84"))
        menu.add_cascade(label='Help', menu=self.HELP)
        self.root.config(menu=menu)


        # -----------RIGHT CLICK POP UP------------------------------------------------------

        self.men = Menu(self.root, tearoff=False)
        self.men.add_command(label='Clear OUTPUT',command=self.clcOutput)
        self.men.add_command(label='Clear Table',command=self.clcTable)

        self.men.add_separator()
        self.men.add_command(label='Show Graphs',command=self.show_graphs)
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
        
        self.__epochs_csv=IntVar()
        ttk.Label(F1,text='Epochs').grid(row=0,column=0,pady=5,padx=5)
        ttk.Entry(F1,width=20,textvariable=self.__epochs_csv).grid(row=0,column=1,pady=5,padx=5)


        # BUTTONS
        ttk.Button(F1,text = 'Open File',width=20,style='info.TButton',
        command=self.File_open).grid(row=0,column=2,pady=5,padx=5)


        self.__b0 = ttk.Button(F1,text = 'Start Training',width=20,style='success.TButton',
        command=self.start_training_csv)
        self.__b0.grid(row=0,column=3,pady=5,padx=5)
        self.__b0['state'] = 'disabled'
        self.__b1 = ttk.Button(F1,text = 'Save the Model',width=20,style='success.Outline.TButton',
        command=self.SaveCsvModel)
        self.__b1.grid(row=0,column=4,pady=5,padx=5)
        self.__b1['state'] = 'disabled'
        self.__b2 = ttk.Button(F1,text = 'Generate Stacked Models',width=20,style='info.Outline.TButton',
        command=self.__StackedModel)
        self.__b2.grid(row=0,column=5,pady=5,padx=5)
        self.__b2['state'] = 'disabled'

        
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
        ttk.Entry(image_frame,width=15,textvariable=self.imgEpoch).grid(row=0,column=3,pady=5,padx=5)

        self.lr= DoubleVar()
        ttk.Label(image_frame,text='Learning Rate').grid(row=0,column=0,pady=5,padx=5)
        ttk.Entry(image_frame,width=10,textvariable=self.lr).grid(row=0,column=1,pady=5,padx=5)
        self.lr.set(3e-4)
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


        self.nam = ttk.Combobox(F1)
        self.nam.grid(row=0,column=7)
        self.nam.set('Select Test Label')
        self.nam['state']='readonly'

        self.pb2 = ttk.Progressbar(image_frame, value=0,length=1280,mode='indeterminate',
         style='success.Horizontal.TProgressbar')
        
        self.pb1 = ttk.Progressbar(F1, value=0,length=1280,mode='indeterminate',
         style='success.Horizontal.TProgressbar')

        self.input_shape = StringVar()
        ttk.Label(image_frame,text='Enter image shape').grid(row=2,column=7)
        ttk.Entry(image_frame,width=20,textvariable=self.input_shape).grid(row=2,column=8,padx=5,pady=5)
        self.input_shape.set('28x28')
        
        

        self.channels = IntVar()
        self.numclass = IntVar()
        ttk.Label(image_frame,text="Enter number of Channels").grid(row=2,column=0,columnspan=2)
        ttk.Entry(image_frame,width=10,textvariable=self.channels).grid(row=2,column=2)
        ttk.Label(image_frame,text="Enter number of Classes").grid(row=2,column=4)
        ttk.Entry(image_frame,width=20,textvariable=self.numclass).grid(row=2,column=5)
        

        self.display_btn = ttk.Button(image_frame,text='Display Graphs',command=self.show_graphs)
        self.display_btn.grid(row=1,column=3)
        self.display_btn['state']='disabled'
                # for loading any trained cnn model 
        self.load_cnn = ttk.Button(image_frame,text='Load Model',command=self.load_cnn_model,width=20)
        self.load_cnn.grid(row=1,column=4)

        ttk.Button(image_frame,text="Open Test Folder",style="info.TButton",command=self.__open_testFolder,
        width=20,).grid(row=2,column=6,padx=5,pady=5)
        
        self.pred_btn=ttk.Button(image_frame,text='Predict',command=self.doPrediction,
        width=20,style='success.TButton')
        self.pred_btn.grid(row=0,column=8,padx=5,pady=5)
        self.pred_btn['state']='disabled'

        self.disp = Text(image_frame,height=6,width=130,background='black',foreground='lime')
        self.disp.grid(row=5,column=0,columnspan=7)

        self.disp1 = Text(image_frame,height=6,width=40,background='black',foreground='yellow')
        self.disp1.grid(row=5,column=7,columnspan=4)

        self.aug_btn = ttk.Button(image_frame,text='Augment Dataset',command=self.__augment,width=20)
        self.aug_btn.grid(row=1,column=8,pady=5,padx=5)
        self.aug_btn['state']='disabled'
        # ----combobox------------
        self.batch_sizes = ttk.Combobox(image_frame,values=[2**i for i in range(9)])
        self.batch_sizes.grid(row=0,column=7)
        self.batch_sizes.set('Select batch size')
        self.batch_sizes['state']='readonly'

        ttk.Label(self.root,text='OUTPUT').pack()
        self.textBox = Text(self.root,height=17,width=180)
        self.textBox.pack()
        self.imgTestdir = None # test image path variable
        self.clockwid = ttk.Label(image_frame)
        self.clockwid.grid(row=1,column=0)
        info = DeviceInfo()
        x_ =ttk.Label(self.root,text='', font=("Arial", 9),foreground='yellow')
        x_.pack(side='right')
        ttk.Label(self.root,text= u"  \u00A9  "+ 'AutoNN', font=("Arial", 9)).pack(side='left')
        sys.stdout = TerminalOutput(self.textBox)
        usage = threading.Thread(target=App._usages,args=(info,self.disp1,x_))
        usage.start()
        root.protocol("WM_DELETE_WINDOW", self.on_closing)


    

    @staticmethod
    def _usages(obj,widget,w2):
        global run 
        # w2 will display the CPU and MEM usage
        w2.config(text=obj.getusage)
        # widget will be used for GPU usage
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
            widget.config(text='{} seconds'.format(clock))
            widget.after(1000,lambda :App.Timer(widget,clock))
        else:
            return


    def __path_settings(self):
        with open(PATH2JSON, "r+") as f:
            data = json.load(f)

        def updatecsvpath():
            __x = filedialog.askdirectory(title='Select Path')
            data['path_csv_models']=__x
            l1.config(text=__x)
            with open(PATH2JSON, "w") as f:
                json.dump(data,f)
            messagebox.showinfo('SAVED',"Saved at {__x}")
            topwindow.destroy()
    
        def updatecnnpath():
            _x = filedialog.askdirectory(title='Select Path')
            data['path_cnn_models']= _x
            l2.config(text=_x)
            with open(PATH2JSON, "w") as f:
                json.dump(data,f)
            messagebox.showinfo('SAVED',"Saved at {_x}")
            topwindow.destroy()
    

        topwindow= Toplevel(self.root)
        topwindow.geometry('500x200')
        ttk.Label(topwindow,text='CNN Models Path: ').pack()
        l1 = ttk.Label(topwindow,text=data['path_cnn_models'],width=50)
        l1.pack()
        ttk.Button(topwindow,text='Update' if data['path_cnn_models'] else 'Add',width=20,
        style='danger.TButton',command= updatecnnpath).pack(padx=5,pady=7)

        ttk.Label(topwindow,text='ANN models Path: ').pack()
        l2 = ttk.Label(topwindow,text=data['path_csv_models'])
        l2.pack()
        ttk.Button(topwindow,text='Update' if data['path_csv_models'] else 'Add',width=20,
        command= updatecsvpath).pack(padx=5,pady=7)
        


    def __augment(self):
        folder = filedialog.askdirectory()
        t0 = threading.Thread(target=self.__augmentation,args=(folder,))
        t0.start()
    
    def __augmentation(self,folder):
        inst = Augment(folder)
        inst.augment()
        inst.get_info()
        messagebox.showinfo('Operation Completed',f"Dataset at path {folder}\n has been augmented")
        return  


    def doPrediction(self):
        filenames = filedialog.askopenfilenames()
        print('Image Files selected are:\n',filenames)
        outputs = self.new_model.predict(paths=filenames)
        print('Prediction vactor is: ',outputs)
        

    def load_cnn_model(self):
        
        path = filedialog.askopenfilename(title='Select Trained Model file',
        filetypes=(('Model files','*.pth'),('model files','*.pt'),
        ('All files','*.*'))
        )
        channels = self.channels.get()
        classes = self.numclass.get()
        if channels <=0 or type(channels)!=int or classes<=0 or type(channels)!=int:
            messagebox.showerror('Invalid Input','Classes or Channels should be\n a NON-Zero integer')
            return
        try:
            self.new_model = CNN(channels,classes)
            self.new_model.load(PATH=path,printmodel=True)
            self.new_model.summary((channels,*tuple(map(int,(self.input_shape.get()).split('x')))))
            self.pred_btn['state']='normal'
            
        except FileNotFoundError:
            self.new_model = resnet(-1,in_channels=channels,num_residual_block=[0,1],num_class=classes)
            self.new_model.load_model(PATH=path)
            self.new_model.summary((channels,*tuple(map(int,(self.input_shape.get()).split('x')))))
            self.pred_btn['state']='normal'

        except Exception as e:
            messagebox.showerror('Invalid Input',e)

    # -------Progress bar FOR IMAGE TRAINING________________
    def get_img_dataset(self):
        self.folder = filedialog.askdirectory(title='Open Folder')
        self.gen_cnn_object = CreateCNN()
        self.show_all_configurations()

    def show_all_configurations(self):
        try:
            self.disp.configure(state="normal")
            self.disp.delete('1.0',END)
            self.disp.insert(END,f'''
            Training Set Path:  {self.folder}             Epochs:     {self.imgEpoch.get()}
            Split Required:     {self.split.get()}        Batch Size: {self.batch_sizes.get()}
            Learning Rate:      {self.lr.get()}           Test Set path: {self.imgTestdir} 
                ''')
            self.disp.configure(state="disabled")

        except Exception as e:
            messagebox.showerror('Empty Path',e)
    

    def __open_testFolder(self):
        self.imgTestdir = filedialog.askdirectory(title='Select Testset Path')
            
    def __Func(self):
        global timeVar
        timeVar=True
        self.pb2.grid(row=4,column=0,columnspan=20)
        self.pb2.start()
        try:
            t=(28,28)
            if self.input_shape.get():
                t=tuple(map(int,(self.input_shape.get()).split('x')))
            path = None  
            try:
                if (self.split.get() is False) and self.imgTestdir:
                    path = self.imgTestdir
            except AttributeError:
                messagebox.showerror('AttributeError',"Make sure testset path is selected\n Hint: Press Open Test Folder")
                return
            self.cnn_model,bestconfig,self.history =self.gen_cnn_object.get_bestCNN(
            path_trainset=self.folder,
            path_testset=path,
            split_required=self.split.get(), 
            batch_size=int(self.batch_sizes.get()), 
            EPOCHS= self.imgEpoch.get(),
            LR=self.lr.get(),
            image_shape=t
            )
            self.display_btn['state']='normal'
            self.savecnn_btn['state']='normal'
            print('Trainig Completed!')
            print('Best Model :\n',self.cnn_model)
            print('History List: ',self.history)
            print('Best Configuration architecture: ',bestconfig)
        except Exception as e:
            messagebox.showerror('Error encountered',e)
            return
        self.pb2.stop()
        self.pb2.grid_remove()
        timeVar = False
        return

        
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
        def save(x,path,shape):
            
            self.cnn_model.save(classes=self.gen_cnn_object.get_classes,
                        image_shape=shape,
                        path=path,filename=x)
            messagebox.showinfo('Model Saved',
            f'Model saved at location "{path}/{x}.pth"')
            pop_model_save_window.destroy()
        
        shape = tuple(map(int,(self.input_shape.get()).split('x')))
        with open(PATH2JSON) as f:
            data = json.load(f)
            if data['path_cnn_models'] == '':
                data['path_cnn_models'] =filedialog.askdirectory(title='Select Path')
            path = data['path_cnn_models']

        pop_model_save_window = Toplevel(self.root)
        pop_model_save_window.geometry('300x100')
        name = StringVar()
        ttk.Label(pop_model_save_window, text = 'Save Model as').pack()
        ttk.Entry(pop_model_save_window,textvariable=name).pack()
        ttk.Button(pop_model_save_window,text="SAVE",width=18,
        command = lambda :save(name.get(),path,shape)).pack(padx=5,pady=5)
        

    # -----------------methods to control csv datasets------------------

    def __start_training_csv(self,a,b,epochs,SAVEHERE):
        try:
            self._atonn = Autonn(a, b, epochs = epochs,save_path = SAVEHERE)
            self._atonn.preprocessing()
            self._atonn.neuralnetworkgeneration()
        except Exception as e:
            messagebox.showerror('Error encountered',e)
            return  
        self.pb1.stop()
        self.pb1.grid_remove()
        self.__b1['state']='normal'
        return 

    def start_training_csv(self):
        with open(PATH2JSON) as f:
            data = json.load(f)
        if data['path_csv_models']:
            _path = data['path_csv_models']
        else: 
            _path = filedialog.askdirectory(title = "Select Model save path")
            data['path_csv_models'] = _path
        
        with open(PATH2JSON, "w") as f:
            json.dump(data, f)  

        self.pb1.start()
        self.pb1.grid(row=3,column=0,columnspan=20)
        if self.__epochs_csv.get() >0 and self.nam.get():
            p1 = threading.Thread(target=self.__start_training_csv,args=(self.csv_file,
                self.nam.get(),self.__epochs_csv.get(),_path))
            p1.start()
        else:    
            messagebox.showerror('Invalid Input','Number of Epochs (int)\n should be  >0')

    def SaveCsvModel(self):
        global switch
        if switch:
            self._atonn.save_candidate_models()
            switch = False
            messagebox.showinfo('Info','MODELS SAVED')
        else:
            self._atonn.save_stacked_models()
            switch = True
            messagebox.showinfo('Info','STACKED MODELS SAVED')
            
        
        self.__b2['state'] = 'normal'

    
    def __StackedModel(self):
        self.pb1.start()
        stackedThread = threading.Thread(target=self.__gen_StackedModel)
        stackedThread.start()

    
    def __gen_StackedModel(self):
        print("Model Stacking STARTED")
        self._atonn.get_stacked_models()
        self.pb1.stop()
        self.pb1.grid_remove()
        messagebox.showinfo('Info','Model Stacking DONE')
        return
    

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
            
        except Exception:   
                messagebox.showerror('ERROR!','Invalid File! Unable to open file!') 

        if self.csv_file:
            self.tree['column']=list(df.columns)
            self.nam.config(values=list(df.columns))
            self.tree['show']='headings'
            for column in self.tree['column']:
                self.tree.column(column,width=90,minwidth=100,stretch=False)
                self.tree.heading(column,text=column)
            self.tree.update()    
            df_rows = df.to_numpy().tolist()
            for row in df_rows:
                self.tree.insert('','end',values=row)

        self.__b0['state'] = 'normal'
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
    App(win,'AutoNN GUI','1280x720')
    win.mainloop()

if __name__=='__main__':
    main()