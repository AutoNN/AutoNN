from matplotlib import pyplot as plt

def plot_graph(history):
    for model in history:
        epochs = len(history[model]['trainloss'])

        fig = plt.figure()
        plt.plot(range(epochs),history[model]['trainloss'],label = 'Train Loss')
        plt.plot(range(epochs),history[model]['valloss'],label= 'Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f"Loss performance of model {model}")
        plt.legend()
        plt.show()

        fig = plt.figure()
        plt.plot(range(epochs),history[model]['trainacc'],label = 'Train Accuracy')
        plt.plot(range(epochs),history[model]['valacc'],label= 'Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f"Accuracy performance of model {model}")
        plt.legend()
        plt.show()
