from autoviz.AutoViz_Class import AutoViz_Class
%matplotlib inline
AV = AutoViz_Class()

filename = 'E:\Seminar Conferences\CDAC\AUTO ML\AutoViz-master\car_design.csv'
class dataviz:
    def __init__(self, name):
        self.name = name
    
    # Viz Method
    def autoviz(self):
        viz = AV.AutoViz(self.name, depVar = "price", chart_format="html") 
 
p = dataviz(filename)
p.autoviz()
