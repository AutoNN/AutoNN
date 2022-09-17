from AutoNN.CNN.cnn_generator import CreateCNN

pop = CreateCNN(5)
best_acc,model,bestconfig,history = pop.get_bestCNN('D:/Personal/dac c dac/autoML/ASC-ML/dataset',split_required=True,EPOCHS=5)
# pop.print_all_cnn_configs()

# model.save()
model.summary((3,28,28))

print(model)
print(best_acc)
print(f'best config {bestconfig}')
print(f'history {history}')
    
