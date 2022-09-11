from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

from model.CNNCodeDuplExtr import CNNCodeDuplExt
from model.data_model import CodeSnippetsDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# Writer will output to ./runs/ directory by default
writer = SummaryWriter()
model=CNNCodeDuplExt()
model=model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

csv_path_neg="data/train/negatives_80p.csv"
csv_path_pos="data/train/positives_80p.csv"
codeSnippetDs = CodeSnippetsDataset([csv_path_pos,csv_path_neg])
codeSnippetDs.x_train.to(device)
codeSnippetDs.y_train.to(device)
train_loader =  DataLoader(codeSnippetDs,batch_size=256,shuffle=True);
csv_path_neg_test="data/test/negatives_20p.csv"
csv_path_pos_test="data/test/positives_20p.csv"
codeSnippetDs_test = CodeSnippetsDataset([csv_path_pos_test,csv_path_neg_test])
codeSnippetDs_test.x_train.to(device)
codeSnippetDs_test.y_train.to(device)
test_loader =  DataLoader(codeSnippetDs_test,batch_size=256,shuffle=True);

for epoch in range(30000):
  running_loss = 0.0
  running_acc = 0.0
  running_loss_test=0.0
  running_acc_test = 0.0

  for i, (train, test) in enumerate(zip(train_loader,test_loader)):
    train_data,y_train=train
    test_data,y_test=test
    test_data= test_data.to(device)
    train_data= train_data.to(device)

    y_train=y_train.to(device)
    y_test=y_test.to(device)


    optimizer.zero_grad()
    y_pred= model(train_data)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    y_pred_test = model(test_data)
    loss_test = criterion(y_pred_test, y_test)
    # print statistics
    running_loss += loss.item()
    running_loss_test+=loss_test.item()
    running_acc += model.calc_accuracy(y_pred,y_train)
    running_acc_test+=model.calc_accuracy(y_pred_test,y_test)
    if i % 10 == 0:  # print every 100 mini-batches
      print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
      print(f'[{epoch + 1}, {i + 1:5d}] accuracy: {running_acc / 10:.3f}')
      writer.add_scalar('Loss/train',running_loss / 10, epoch + 1)
      writer.add_scalar('Accuracy/train', running_acc / 10, epoch + 1)
      writer.add_scalar('Loss/test', running_loss_test / 10, epoch + 1)
      writer.add_scalar('Accuracy/test', running_acc_test / 10, epoch + 1)
      running_loss = 0.0
      running_acc = 0.0
      running_loss_test = 0.0
      running_acc_test = 0.0




