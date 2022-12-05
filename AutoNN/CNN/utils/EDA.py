from matplotlib.pyplot import (plot,legend,show,figure,ylabel,xlabel,title)

def plot_graph(history):
    for model in history:
        epochs = len(history[model]['trainloss'])

        fig = figure()
        plot(range(epochs),history[model]['trainloss'],label = 'Train Loss')
        plot(range(epochs),history[model]['valloss'],label= 'Validation Loss')
        xlabel('Epochs')
        ylabel('Loss')
        title(f"Loss performance of model {model}")
        legend()
        show()

        fig = figure()
        plot(range(epochs),history[model]['trainacc'],label = 'Train Accuracy')
        plot(range(epochs),history[model]['valacc'],label= 'Validation Accuracy')
        xlabel('Epochs')
        ylabel('Accuracy')
        title(f"Accuracy performance of model {model}")
        legend()
        show()
