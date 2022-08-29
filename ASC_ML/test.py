from CNN.cnn_generator import Population,create_config

# from pytorchsummary import summary
# l = random.randint(3,10)
# model = CNN(3,create_config())
# summary((3,32,32),model)
# print(model)
    

pop = Population(10,3,4)
# pop.print_all_architecture()
best_acc,model = pop.get_bestCNN('dataset',split_required=True)
print(model)
    
