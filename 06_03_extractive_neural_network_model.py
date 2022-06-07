# #########################################################
# 0.0 Import
# #########################################################
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from Project.functions import return_df_pred_summaries, rouge_scores


# #########################################################
# 1.0 Load dataset
# #########################################################
train_prep = pickle.load(open(".//Project/interim/train_prep.pickle", "rb"))
test_prep = pickle.load(open(".//Project/interim/test_prep.pickle", "rb"))
validation_prep = pickle.load(open(".//Project/interim/validation_prep.pickle", "rb"))

# select relevant datasets
train_X = train_prep["df_X"].drop(["sent_number", "doc_Length"], axis=1)
train_y = train_prep["y_array"].ravel()
validation_X = validation_prep["df_X"].drop(["sent_number", "doc_Length"], axis=1)
validation_y = validation_prep["y_array"].ravel()
test_X = test_prep["df_X"].drop(["sent_number", "doc_Length"], axis=1)
test_y = test_prep["y_array"].ravel()
test_doc_label = test_prep["Xy_doc_label_array"]
test_article_clean = test_prep["df_original"]["article_clean_spacy"]
test_highlight_clean = test_prep["df_original"]["highlights_clean_spacy"]

validation_doc_label = validation_prep["Xy_doc_label_array"]
validation_article_clean = validation_prep["df_original"]["article_clean_spacy"]
validation_highlight_clean = validation_prep["df_original"]["highlights_clean_spacy"]

# remove redundant datasets
del train_prep
del test_prep
del validation_prep
# #########################################################


# #########################################################
# 2.0 Neural Network
# #########################################################

# train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return torch.FloatTensor(self.X_data.iloc[index]), torch.FloatTensor([self.y_data[index]])

    def __len__(self):
        return len(self.X_data)


train_data = TrainData(train_X,
                       train_y)


# validation data
class ValidationData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        self.X_data_tensor = torch.FloatTensor(X_data.values)

    def __getitem__(self, index):
        return torch.FloatTensor(self.X_data.iloc[index]), torch.FloatTensor([self.y_data[index]])

    def __len__(self):
        return len(self.X_data)


validation_data = ValidationData(validation_X, validation_y)


# test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = torch.FloatTensor(X_data.values)

    def __getitem__(self, index):
        return torch.FloatTensor(self.X_data.iloc[index])

    def __len__(self):
        return len(self.X_data)


test_data = TestData(test_X)

train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
validation_loader = DataLoader(dataset=validation_data, batch_size=128)
test_loader = DataLoader(dataset=test_data)


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()

        self.layer_1 = nn.Linear(1536, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return (x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = BinaryClassification().to(device)
print(model)



class_count = Counter(train_y)
class_weights = torch.Tensor([len(train_y) / c for c in pd.Series(class_count).sort_index().values])
# Cant iterate over class_count because dictionary is unordered
class_weights = class_weights.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([(np.shape(train_y)[0] / sum(train_y))]).to(device))
optimizer = optim.SGD(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


def binary_acc(y_pred, y_validation):
    acc = balanced_accuracy_score(y_validation, y_pred)
    return acc


def train_epoch(model, train_loader, loss_fn, optimizer, GPU=True):
    # Variables to keep track of the loss and accuracy:
    running_loss = 0
    running_acc = 0
    for i, batch in enumerate(train_loader):
        # Loading a batch of training data (using GPU, if available):
        input, target = batch
        if GPU:
            input, target = input.to(device), target.to(device)

        optimizer.zero_grad()  # Deleting leftover gradients from the optimizer

        # Predicting and calculating loss:
        pred = model(input)

        loss = loss_fn(pred, target)

        running_loss += loss.item()
        # Backpropagate the loss through the network and make a step with the optimizer:
        loss.backward()
        optimizer.step()

        running_acc += binary_acc(torch.round(torch.sigmoid(pred)).cpu().detach().numpy().flatten(),
                                  target.cpu().numpy().flatten())

        print(f"batch {i} balanced_accuracy: {running_acc / (i + 1)} loss: {running_loss / (i + 1)}")

    return running_loss / (i + 1), running_acc / (i + 1)


# Complete training function:

def train(model, train_loader, val_loader, loss_fn, optimizer, epochs=100, GPU=False):
    # Dictionaries to note down performance in each epoch:
    loss_lists = {'train': [], 'val': []}
    acc_lists = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train(True)  # Tells the model that we will need gradients in the following part
        epoch_loss, epoch_acc = train_epoch(model, train_loader, loss_fn, optimizer=optimizer, GPU=GPU)

        loss_lists['train'].append(epoch_loss)
        acc_lists['train'].append(epoch_acc)

        print(f"Train loss in epoch {epoch}: {epoch_loss}")
        print(f"Train accuracy in epoch {epoch}: {epoch_acc}")

        model.train(False)  # Tells the model that we will NOT need gradients in the following part

        # Validation loop:
        # (Basically the same as the training function for one epoch, but without gradient descent)
        running_vloss = 0
        running_vacc = 0
        model.eval()
        for i, vbatch in enumerate(val_loader):
            vinput, vtarget = vbatch
            if GPU:
                vinput, vtarget = vinput.to(device), vtarget.to(device)

            vpred = model(vinput)
            vloss = loss_fn(vpred, vtarget)
            running_vloss += vloss.item()
            running_vacc += binary_acc(torch.round(torch.sigmoid(vpred)).cpu().detach().numpy().flatten(),
                                       vtarget.cpu().numpy().flatten())
        epoch_vloss = running_vloss / (i + 1)
        epoch_vacc = running_vacc / (i + 1)

        loss_lists['val'].append(epoch_vloss)
        acc_lists['val'].append(epoch_vacc)

        print(f"Validation loss in epoch {epoch}: {epoch_vloss}")
        print(f"Validation accuracy in epoch {epoch}: {epoch_vacc}", '\n')

    return loss_lists, acc_lists


losses1, accs1 = train(model, train_loader, validation_loader, criterion, optimizer=optimizer, epochs=10, GPU=True)

losses2, accs2 = train(model, train_loader, validation_loader, criterion, optimizer=optimizer, epochs=10, GPU=True)

losses3, accs3 = train(model, train_loader, validation_loader, criterion, optimizer=optimizer, epochs=10, GPU=True)

losses_full = {"train": losses1["train"] + losses2["train"] + losses3["train"],
               "val": losses1["val"] + losses2["val"] + losses3["val"]}
accs_full = {"train": accs1["train"] + accs2["train"] + accs3["train"],
             "val": accs1["val"] + accs2["val"] + accs3["val"]}

NN_plot_data = [losses_full, accs_full]
pickle.dump(NN_plot_data, open("./Project/interim/NN_torch_30_plot.pickle", "wb"))

NN_plot_data = pickle.load(open("./Project/interim/NN_torch_30_plot.pickle", "rb"))


NN_plot_data


fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].plot(NN_plot_data[0]['train'], label='train')
axes[0].plot(NN_plot_data[0]['val'], label='validation')
axes[0].set_title('Losses')
axes[0].set_xlabel('Epochs')
axes[0].legend()

accuracies_train = [i.item() for i in NN_plot_data[1]["train"]]
accuracies_val = [i.item() for i in NN_plot_data[1]["val"]]

axes[1].plot(accuracies_train, label='train')
axes[1].plot(accuracies_val, label='validation')
axes[1].set_title('Balanced accuracy')
axes[1].set_xlabel('Epochs')
axes[1].legend()

plt.show()

torch.save(model, "./Project/interim/NN_torch_30.pth")

with torch.no_grad():
    model_torch = torch.load("./Project/interim/NN_torch_30.pth")
    model_torch.eval()

    vpred = torch.sigmoid(model_torch(test_data.X_data.to("cuda"))).to("cpu")

test_prob = vpred
test_pred = np.array(torch.round(vpred)).flatten()
# #########################################################


# #########################################################
# 4.0 Backtest model
# #########################################################
cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
acc = accuracy_score(test_y, test_pred)
balanced_acc = balanced_accuracy_score(test_y, test_pred)
f1 = f1_score(test_y, test_pred)

idx, doc_index, pred_summaries = return_df_pred_summaries(Xy_doc_label=test_doc_label, y_pred=test_pred,
                                                          df_text=test_article_clean,
                                                          thresh=0.5,
                                                          min_num=1)


df_gold = test_highlight_clean[doc_index]
gold_summaries = [' '.join(df_gold.iloc[j]) for j in range(len(pred_summaries))]
summaries_comp = tuple(zip(pred_summaries, gold_summaries))

scores = rouge_scores(pred_summaries, gold_summaries)

results_dict = {'conf_matrix': cm, 'accuracy': acc, 'balanced_acc': balanced_acc, 'f1': f1,
                'summaries_comp': summaries_comp,
                'sent_index_number': idx, 'Rouge': scores}

pickle.dump(results_dict, open("./Project/output/NN_results.pickle", "wb"))
# #########################################################


# #########################################################
# 4.1 Backtest model - best 4
# #########################################################
test_prediction_data = pd.DataFrame({"label": test_doc_label.flatten(), "pred": test_prob.to("cpu").numpy().flatten()})
test_prediction_data["forth"] = test_prediction_data.groupby("label")["pred"].transform(lambda x: x.nlargest(4).min())
test_prediction_data["pred_best4"] = np.where((test_prediction_data["pred"] >= test_prediction_data["forth"]), 1, 0)

test_pred = test_prediction_data["pred_best4"].values

cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
acc = accuracy_score(test_y, test_pred)
balanced_acc = balanced_accuracy_score(test_y, test_pred)
f1 = f1_score(test_y, test_pred)

idx_4, doc_index_4, pred_summaries_4 = return_df_pred_summaries(Xy_doc_label=test_doc_label, y_pred=test_pred,
                                                                df_text=test_article_clean,
                                                                thresh=0.5,
                                                                min_num=1)

df_gold_4 = test_highlight_clean[doc_index_4]
gold_summaries_4 = [' '.join(df_gold_4.iloc[j]) for j in range(len(pred_summaries_4))]
summaries_comp_4 = tuple(zip(pred_summaries_4, gold_summaries_4))

scores = rouge_scores(pred_summaries_4, gold_summaries_4)

results_dict_nn_4 = {'conf_matrix': cm, 'accuracy': acc, 'balanced_acc': balanced_acc, 'f1': f1,
                    'summaries_comp': summaries_comp_4,
                    'sent_index_number': idx_4, 'Rouge': scores}

pickle.dump(results_dict_nn_4, open("./Project/output/NN_4_results.pickle", "wb"))
# #########################################################


# keras implementation
# import keras as keras
# from keras.models import Sequential
# from keras.layers import Dense
# import tensorflow as tf
#
# tf.compat.v1.disable_eager_execution()
# # class_weights for imbalanced data
# pos_w = int(train_y.shape[0] / sum(train_y == 1))
# weight_dict = {0: 1, 1: pos_w / 2}
#
# # Define Model
# model = Sequential()
# model.add(Dense(64, input_dim=1536, activation='relu'))
# # model.add(Dense(50, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# # Compile Model
# model.compile(loss='binary_crossentropy', optimizer='adam',
#               metrics=['accuracy'])
#
#
# class Metrics(keras.callbacks.Callback):
#
#     def __init__(self, val_data, batch_size=20):
#         super().__init__()
#         self.validation_data = val_data
#         self.batch_size = batch_size
#
#     def on_train_begin(self, logs={}):
#         self._data = []
#
#     def on_epoch_end(self, batch, logs={}):
#         X_val, y_val = self.validation_data[0], self.validation_data[1]
#         y_predict = (np.asarray(model.predict(X_val)).flatten() > 0.5)
#
#         self._data.append({
#             'val_balanced_acc': balanced_accuracy_score(y_val, y_predict),
#         })
#         return
#
#     def get_data(self):
#         return self._data
#
#
# # Fit Model
# callback_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# callback_metric = Metrics(val_data=[validation_X, validation_y])
#
# history = model.fit(train_X, train_y, validation_data=[validation_X, validation_y], epochs=20, batch_size=128,
#                     callbacks=[callback_early, callback_metric], class_weight=weight_dict)  # class_weight=weight_dict
#
# # same metrics as torch
# callback_metric.get_data()
#
# # Predict Model
# y_pred = model.predict(test_X)
