from pprint import pprint
import json

from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, f1_score
from model.CNNCodeDuplExtr import CNNCodeDuplExt
from model.ModelFactory import ModelFactory
from model.data_model import CodeSnippetsDataset
from sklearn.metrics import precision_recall_curve,precision_score
from sklearn.metrics import auc
import warnings
warnings.filterwarnings("ignore")
def train_cnn_model(model,nr_epochs,train_loader,test_loader,verbose=False):
  # Writer will output to ./runs/ directory by default
  writer = SummaryWriter()
  model = model.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  for epoch in range(nr_epochs):
    running_loss = 0.0
    running_acc = 0.0
    running_loss_test = 0.0
    running_acc_test = 0.0

    for i, (train, test) in enumerate(zip(train_loader, test_loader)):
      train_data, y_train = train
      test_data, y_test = test
      test_data = test_data.to(device)
      train_data = train_data.to(device)

      y_train = y_train.to(device)
      y_test = y_test.to(device)

      optimizer.zero_grad()
      y_pred = model(train_data)

      # Compute and print loss
      loss = criterion(y_pred, y_train)
      loss.backward()
      optimizer.step()
      y_pred_test = model(test_data)
      loss_test = criterion(y_pred_test, y_test)
      # print statistics
      running_loss += loss.item()
      running_loss_test += loss_test.item()
      running_acc += model.calc_accuracy(y_pred, y_train)
      running_acc_test += model.calc_accuracy(y_pred_test, y_test)
      if i % 10 == 0:  # print every 100 mini-batches
        if verbose:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            print(f'[{epoch + 1}, {i + 1:5d}] accuracy: {running_acc / 10:.3f}')
        writer.add_scalar('Loss/train', running_loss / 10, epoch + 1)
        writer.add_scalar('Accuracy/train', running_acc / 10, epoch + 1)
        writer.add_scalar('Loss/test', running_loss_test / 10, epoch + 1)
        writer.add_scalar('Accuracy/test', running_acc_test / 10, epoch + 1)
        running_loss = 0.0
        running_acc = 0.0
        running_loss_test = 0.0
        running_acc_test = 0.0
  return model
def calc_metrics(y_pred_test,y_true_test):

    accuracy = accuracy_score(y_true_test, y_pred_test)
    precision, recall, thresholds = precision_recall_curve(y_true_test, y_pred_test)
    precision_sc = precision_score(y_true_test, y_pred_test, average='weighted')
    recall_sc = recall_score(y_true_test, y_pred_test, average='weighted')
    area_under_the_curve = auc(recall, precision)
    f1_sc = f1_score(y_true_test, y_pred_test, average='weighted')
    metrics= {'accuracy':accuracy,'precision':precision_sc,'recall':recall_sc,'auc':area_under_the_curve,"f1_sc":f1_sc}
    for k in metrics.keys():
        metrics[k] = round(metrics[k], 2)
    return metrics

def train_sklearn(model,train_data):
  X,y=train_data
  model.fit(X,y)
  return model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)
names_models={'rf': 'Random Forest',
'sgd': 'Support Vector Machine',
'gnb': 'Naive Bayes',
'lrc': "Logistic Regression",
'cnn': "Convolutional Neural Network"}
type_to_implementation = (
            'rf',
            'sgd',
            'gnb',
            'lrc',
            'cnn',


)

model_factory=ModelFactory()



csv_path_neg="data/train/negatives_80p.csv"
csv_path_pos="data/train/positives_80p.csv"
codeSnippetDs = CodeSnippetsDataset([csv_path_pos,csv_path_neg])
train_loader =  DataLoader(codeSnippetDs,batch_size=256,shuffle=True);
csv_path_neg_test="data/test/negatives_20p.csv"
csv_path_pos_test="data/test/positives_20p.csv"
codeSnippetDs_test = CodeSnippetsDataset([csv_path_pos_test,csv_path_neg_test])
test_loader =  DataLoader(codeSnippetDs_test,batch_size=256,shuffle=True);
train_results={}
nr_epochs=500
for  model_type in type_to_implementation:
  model = model_factory.make_model(model_type)
  print(model_type)

  if model_type=="cnn":
      model=train_cnn_model(model,nr_epochs,train_loader,test_loader,verbose=False)
      X_test, y_true_test = codeSnippetDs_test.get_all_data(model_type=model_type)
      X_test=X_test.to(device)
      y_pred_test = model(X_test)
      y_pred_test = torch.argmax(y_pred_test,-1)
      metrics=calc_metrics(y_pred_test.cpu().detach().numpy(), y_true_test.cpu().detach().numpy())


  else:

      model=train_sklearn(model,codeSnippetDs.get_all_data(model_type=model_type))
      X_test, y_true_test = codeSnippetDs_test.get_all_data(model_type=model_type)
      y_pred_test = model.predict(X_test)
      metrics=calc_metrics(y_pred_test,y_true_test)
  pprint(metrics)

  train_results[names_models[model_type]]=metrics


with open('train_results_{}_epochs.json'.format(str(nr_epochs)), 'w') as fp:
    json.dump(train_results, fp)


