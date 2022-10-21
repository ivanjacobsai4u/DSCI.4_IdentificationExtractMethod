import logging
import warnings
warnings.filterwarnings("ignore")
from ModelTrainer import ModelTrainer
import json
import numpy as np
from mlxtend.evaluate import mcnemar,mcnemar_table
import itertools
import scipy.stats as stats

with open('configs/configs.json',"r") as f:
    configs=json.load(f)

names_models=configs['statistics']["names_models"]
type_to_implementation =configs['statistics']["type_to_implementation"]
csv_path_neg=configs['statistics']["csv_path_neg"]
csv_path_pos=configs['statistics']["csv_path_pos"]
csv_path_neg_test=configs['statistics']["csv_path_neg_test"]
csv_path_pos_test=configs['statistics']["csv_path_pos_test"]
nr_epochs=configs['statistics']['nr_epochs']
model_trainer=ModelTrainer(names_models=names_models,
                           type_to_implementation=type_to_implementation,
                           csv_path_neg=csv_path_neg,
                           csv_path_pos=csv_path_pos,
                           csv_path_neg_test=csv_path_neg_test,
                           csv_path_pos_test=csv_path_pos_test)
train_merge_results={}
for model_name in names_models.values():
    train_merge_results[model_name]={'y_pred_test':[],'y_true_test':[]}

model_pairs=list(itertools.combinations(names_models.values(),2))
odds_results={}
for pair in model_pairs:
    model_one, model_two = pair
    odds_results[model_one + '_' + model_two]={'oddsratios':[],'pvalues':[]}

for n in range(configs['statistics']['nr_iterations']):
    _,train_results=model_trainer.train(nr_epochs=nr_epochs,persist_results=False)
    logging.info(train_results)
    for model_name in names_models.values():
        train_merge_results[model_name]['y_pred_test'].extend(train_results[model_name]['y_pred_test'])
        train_merge_results[model_name]['y_true_test'].extend(train_results[model_name]['y_true_test'])

    for pair in model_pairs:
        model_one, model_two = pair
        y_target = np.array(train_results[model_one]['y_true_test'])
        y_model1 = np.array(train_results[model_one]['y_pred_test'])
        y_model2 = np.array(train_results[model_two]['y_pred_test'])
        tb = mcnemar_table(y_target=y_target,
                           y_model1=y_model1,
                           y_model2=y_model2)
        chi2, p = mcnemar(ary=tb, exact=True,corrected=True)
        oddsratio, pvalue = stats.fisher_exact(tb)
        odds_results[model_one + '_' + model_two]['oddsratios'].append(oddsratio)
        odds_results[model_one + '_' + model_two]['pvalues'].append(pvalue)
stats_results={}
for pair in model_pairs:
    model_one,model_two=pair
    y_target=np.array(train_merge_results[model_one]['y_true_test'])
    y_model1=np.array(train_merge_results[model_one]['y_pred_test'])
    y_model2=np.array(train_merge_results[model_two]['y_pred_test'])
    tb=mcnemar_table(y_target=y_target,
                     y_model1=y_model1,
                     y_model2=y_model2)
    chi2,p=mcnemar(ary=tb,corrected=True)
    # oddsratio,pvalue=stats.fisher_exact(tb)
    stats_results[model_one+'_'+model_two]={'model_1':model_one,'model_2':model_two, 'model_pairs':pair,
                                            'chi-squared':chi2,'p-value':round(p,5),
                                            'oddsratio':round(np.min(np.array(odds_results[model_one + '_' + model_two]['oddsratios'])),2),
                                            'oddsratios':odds_results[model_one + '_' + model_two]['oddsratios'],
                                            'pvalue':round(np.min(np.array(odds_results[model_one + '_' + model_two]['pvalues'])),2) }

with open('statistics_results_{}_epochs_{}_iterations.json'.format(str(nr_epochs),
                                                                   str(configs['statistics']['nr_iterations'])), 'w') as fp:
                json.dump(stats_results, fp)

with open('preds_stats_results_{}_epochs_{}_iterations.json'.format(str(nr_epochs),
                                                                                   str(configs['statistics'][
                                                                                           'nr_iterations'])),
                          'w') as fp:
                    json.dump(train_merge_results, fp)

logging.info('finished')
print("finished")