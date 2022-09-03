from ASC_ML.CNN.cnn_generator import CreateCNN

# from torch.multiprocessing import freeze_support

# if __name__ == '__main__':
#     freeze_support()
pop = CreateCNN()
best_acc,model,bestconfig,history = pop.get_bestCNN('dataset',split_required=True,EPOCHS=1)
# pop.print_all_cnn_configs()

model.save()
model.summary((3,28,28))

print(model)
print(best_acc)
print(f'best config {bestconfig}')
print(f'history {history}')
    
