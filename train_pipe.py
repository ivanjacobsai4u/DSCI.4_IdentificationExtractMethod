import logging
import warnings
warnings.filterwarnings("ignore")
from ModelTrainer import ModelTrainer
import json


with open('configs/configs.json',"r") as f:
    configs=json.load(f)

names_models=configs['train']["names_models"]
type_to_implementation =configs['train']["type_to_implementation"]
csv_path_neg=configs['train']["csv_path_neg"]
csv_path_pos=configs['train']["csv_path_pos"]
csv_path_neg_test=configs['train']["csv_path_neg_test"]
csv_path_pos_test=configs['train']["csv_path_pos_test"]
nr_epochs=configs['train']['nr_epochs']
model_trainer=ModelTrainer(names_models=names_models,
                           type_to_implementation=type_to_implementation,
                           csv_path_neg=csv_path_neg,
                           csv_path_pos=csv_path_pos,
                           csv_path_neg_test=csv_path_neg_test,
                           csv_path_pos_test=csv_path_pos_test)
train_results=model_trainer.train(nr_epochs=nr_epochs,persist_results=True)
logging.info(train_results)
logging.info('finished')
