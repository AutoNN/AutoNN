import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def scatter_plot(x,y,xlabel: str,ylabel: str,title: str):
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def ConfusionMatrix(model,num_classes,colourmap='Greens',annotations=True,barLabel = None):
    cm = np.zeros((num_classes, num_classes))


    sns.heatmap(cm,cmap=colourmap,annot=annotations,cbar_kws={"label":barLabel})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')