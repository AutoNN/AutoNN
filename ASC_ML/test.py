from CNN.cnn_generator import CreateCNN

# from torch.multiprocessing import freeze_support

# if __name__ == '__main__':
#     freeze_support()
pop = CreateCNN(3,3,10)
# pop.print_all_cnn_configs()
best_acc,model = pop.get_bestCNN('dataset',split_required=True)
print(model, best_acc)
    
