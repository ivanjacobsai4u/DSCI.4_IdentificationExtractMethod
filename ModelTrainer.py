
from pprint import pprint
import json

from torch.utils.data import DataLoader
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, f1_score
from model.CNNCodeDuplExtr import CNNCodeDuplExt
from model.ModelFactory import ModelFactory
from model.data_model import CodeSnippetsDataset
from sklearn.metrics import precision_recall_curve,precision_score
from sklearn.metrics import auc

class ModelTrainer(object):
    def __init__(self,type_to_implementation,names_models,
                verbosity=logging.INFO,
                csv_path_neg = "data/train/negatives_80p.csv",
                csv_path_pos = "data/train/positives_80p.csv",
                csv_path_neg_test = "data/test/negatives_20p.csv",
                csv_path_pos_test = "data/test/positives_20p.csv" ):
        self.type_to_implementation=type_to_implementation
        self.names_models=names_models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter()
        self.model_factory=ModelFactory()
        self.csv_path_neg = csv_path_neg
        self.csv_path_pos = csv_path_pos
        self.csv_path_neg_test = csv_path_neg_test
        self.csv_path_pos_test = csv_path_pos_test
        self.codeSnippetDs = CodeSnippetsDataset([self.csv_path_pos, self.csv_path_neg])
        self.train_loader = DataLoader(self.codeSnippetDs, batch_size=256, shuffle=True);

        self.codeSnippetDs_test = CodeSnippetsDataset([self.csv_path_pos_test, self.csv_path_neg_test])
        self.test_loader = DataLoader(self.codeSnippetDs_test, batch_size=256, shuffle=True);

        logging.basicConfig(filename='training.log', encoding='utf-8', level=verbosity)
    def _train_cnn_model(self,model, nr_epochs, train_loader, test_loader):
        # Writer will output to ./runs/ directory by default

        model = model.to(self.device)
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
                test_data = test_data.to(self.device)
                train_data = train_data.to(self.device)

                y_train = y_train.to(self.device)
                y_test = y_test.to(self.device)

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

                    logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                    logging.info(f'[{epoch + 1}, {i + 1:5d}] accuracy: {running_acc / 10:.3f}')
                    self.writer.add_scalar('Loss/train', running_loss / 10, epoch + 1)
                    self.writer.add_scalar('Accuracy/train', running_acc / 10, epoch + 1)
                    self.writer.add_scalar('Loss/test', running_loss_test / 10, epoch + 1)
                    self.writer.add_scalar('Accuracy/test', running_acc_test / 10, epoch + 1)
                    running_loss = 0.0
                    running_acc = 0.0
                    running_loss_test = 0.0
                    running_acc_test = 0.0
        return model

    def _calc_metrics(self,y_pred_test, y_true_test):

        accuracy = accuracy_score(y_true_test, y_pred_test)
        precision, recall, thresholds = precision_recall_curve(y_true_test, y_pred_test)
        precision_sc = precision_score(y_true_test, y_pred_test, average='weighted')
        recall_sc = recall_score(y_true_test, y_pred_test, average='weighted')
        area_under_the_curve = auc(recall, precision)
        f1_sc = f1_score(y_true_test, y_pred_test, average='weighted')
        metrics = {'accuracy': accuracy, 'precision': precision_sc, 'recall': recall_sc, 'auc': area_under_the_curve,
                   "f1_sc": f1_sc}
        for k in metrics.keys():
            metrics[k] = round(metrics[k], 2)
        return metrics

    def _train_sklearn(self,model, train_data):
        X, y = train_data
        model.fit(X, y)
        return model
    def train(self,nr_epochs,persist_results=True):
        train_results = {}

        for model_type in self.type_to_implementation:
            model = self.model_factory.make_model(model_type)
            logging.info(model_type)

            if model_type == "cnn":
                model = self._train_cnn_model(model, nr_epochs, self.train_loader, self.test_loader)
                X_test, y_true_test = self.codeSnippetDs_test.get_all_data(model_type=model_type)
                X_test = X_test.to(self.device)
                y_pred_test = model(X_test)
                y_pred_test = torch.argmax(y_pred_test, -1)

                metrics = self._calc_metrics(y_pred_test.cpu().detach().numpy(), y_true_test.cpu().detach().numpy())
                train_results[self.names_models[model_type]] = {'metrics': metrics, 'y_pred_test': y_pred_test.cpu().detach().numpy().tolist(),
                                                                'y_true_test': y_true_test.cpu().detach().numpy().tolist()}
            else:
                model = self._train_sklearn(model, self.codeSnippetDs.get_all_data(model_type=model_type))
                X_test, y_true_test = self.codeSnippetDs_test.get_all_data(model_type=model_type)
                y_pred_test = model.predict(X_test)

                metrics = self._calc_metrics(y_pred_test, y_true_test)

                train_results[self.names_models[model_type]] = {'metrics':metrics,'y_pred_test':y_pred_test.tolist(),'y_true_test':y_true_test.tolist()}
            logging.info(metrics)
        if persist_results:
            with open('train_results_{}_epochs.json'.format(str(nr_epochs)), 'w') as fp:
                json.dump(train_results, fp)

        return train_results
