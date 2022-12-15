__all__ =[   'basic_data_cleaning',
             'dataframe_extractor',
             'dropout_optimization',
             'final',
             'hyperparameter_optimization',
             'model_generation',
             'model_gen_train_test',
             'model_optimization',
             'model_stacking',
             'multiple_model_gen_v1',
             'multiple_model_gen_v2',
             'multiple_model_gen_v3',
             'search_space_gen_v1',
             'utilities']

# def appendallmodules():
#     import os 
#     for i in os.listdir(os.path.dirname(__file__)):
#         if i.endswith('.py') and not i.startswith("__"):
#             __all__.append(i.replace('.py',''))
#     print(__all__)

# appendallmodules()