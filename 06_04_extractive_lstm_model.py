# #########################################################
# 0.0 Import
# #########################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from random import randrange
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score

import keras as keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

import tensorflow as tf

from Project.functions import return_df_pred_summaries, rouge_scores
# #########################################################


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

train_original = train_prep["df_original"]
test_original = test_prep["df_original"]
validation_original = validation_prep["df_original"]


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
# 2.0 Creating the format requirement for the
# TimeDistributed Dense layer
# #########################################################
train_original["article_embedding"] = train_original.article_embedding.apply(lambda x: np.array(x))
train_original["article_embedding"] = train_original.article_embedding.apply(lambda x: x.reshape(1, x.shape[0], x.shape[1]))
train_original["labels"] = train_original.labels.apply(lambda x: np.array(x))
train_original["labels"] = train_original.labels.apply(lambda x: x.reshape(1, len(x), 1))

test_original["article_embedding"] = test_original.article_embedding.apply(lambda x: np.array(x))
test_original["article_embedding"] = test_original.article_embedding.apply(lambda x: x.reshape(1, x.shape[0], x.shape[1]))
test_original["labels"] = test_original.labels.apply(lambda x: np.array(x))
test_original["labels"] = test_original.labels.apply(lambda x: x.reshape(1, len(x), 1))

validation_original["article_embedding"] = validation_original.article_embedding.apply(lambda x: np.array(x))
validation_original["article_embedding"] = validation_original.article_embedding.apply(lambda x: x.reshape(1, x.shape[0], x.shape[1]))
validation_original["labels"] = validation_original.labels.apply(lambda x: np.array(x))
validation_original["labels"] = validation_original.labels.apply(lambda x: x.reshape(1, len(x), 1))


X_train = train_original.article_embedding.tolist()
y_train = train_original.labels.tolist()

X_test = test_original.article_embedding.tolist()
y_test = test_original.labels.tolist()

X_validation = validation_original.article_embedding.tolist()
y_validation = validation_original.labels.tolist()
# #########################################################


# #########################################################
# 3.0 Custom metrics for keras backtesting
# #########################################################
class Metrics(keras.callbacks.Callback):

    def __init__(self, model, train_data, val_data, batch_size=1):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self._data_val = []
        self._data_train = []
        self._data_val_loss = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1][0]
        X_train, y_train = self.train_data[0], self.train_data[1][0]

        y_predict_train = np.zeros(X_train.shape[1])
        y_predict_val = np.zeros(X_val.shape[1])

        idx_train_best4 = np.argsort(self.model.predict(X_train).flatten())[-4:]
        np.put(y_predict_train, idx_train_best4, 1)

        idx_val_best4 = np.argsort(self.model.predict(X_val).flatten())[-4:]
        np.put(y_predict_val, idx_val_best4, 1)

        self._data_val.append(balanced_accuracy_score(y_val, y_predict_val))
        self._data_train.append(balanced_accuracy_score(y_train, y_predict_train))
        self._data_val_loss.append(tf.keras.metrics.binary_crossentropy(y_val, y_predict_val))
        return

    def get_data(self):
        return self._data_val, self._data_train, self._data_val_loss
# #########################################################


# #########################################################
# 4.0 Gated Recurrent Unit
# #########################################################

# define model
model_gru = Sequential()
model_gru.add(Bidirectional(tf.keras.layers.GRU(100, input_shape=(None, 768), return_sequences=True, dropout=0)))
model_gru.add(TimeDistributed(Dense(10, activation='relu')))
model_gru.add(Dense(1, activation='sigmoid'))


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model_gru.compile(loss='binary_crossentropy', optimizer=optimizer,
              metrics=["accuracy"])


# train GRU
training_loss_avg_gru, training_metric_val_gru, training_metric_train_gru, training_metric_val_loss_gru = list(), list(), list(), list()
for epoch in range(10):
    for j in range(len(X_train)):
        training_loss = list()
        X, y = X_train[j], y_train[j]
        val_index = randrange(5000)
        X_val, y_val = X_validation[val_index], y_validation[val_index]
        callback_metric = Metrics(model_gru, train_data=[X, y],val_data=[X_val, y_val])
        history = model_gru.fit(X, y, validation_data=[X_val, y_val], epochs=1, batch_size=1, callbacks=[callback_metric])
        training_loss.append(history.history['loss'])

        if j % 100 == 0:
            print(f"#### {j} sentences finished!#### ")
    print(f" EPOCH finished! ######## {epoch + 1} EPOCH finished! ########")

    training_loss_avg_gru.append(np.mean(training_loss))
    training_metric_val_gru.append(np.mean(callback_metric.get_data()[0]))
    training_metric_train_gru.append(np.mean(callback_metric.get_data()[1]))
    training_metric_val_loss_gru.append(np.mean(callback_metric.get_data()[2]))



fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].plot(training_loss_avg_gru, label='train')
axes[0].plot(training_metric_val_loss_gru, label='validation')
axes[0].set_title('Losses')
axes[0].set_xlabel('Epochs')
axes[0].legend()

axes[1].plot(training_metric_train_gru, label='train')
axes[1].plot(training_metric_val_gru, label='validation')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epochs')
axes[1].legend()

plt.show()


model_gru.summary()
model_gru.save("./Project/interim/model_gru")

result_model_metrics = {"train_loss":training_loss_avg_gru, "train_bacc":training_metric_train_gru,
                        "val_loss": training_metric_val_loss_gru, "val_bacc": training_metric_val_gru}

pickle.dump(result_model_metrics, open("./Project/output/GRU_train_metric_results.pickle", "wb"))



# evaluate GRU
y_pred_list_gru, idx_list_gru, preds = list(), list(), list()
for j in range(len(X_test)):
    X = X_test[j]
    y_pred = model_gru.predict(X, verbose=0)
    idx = np.argsort(y_pred[0].flatten())[-4:]
    idx = sorted(idx)
    y_pred_list_gru.append(y_pred)
    idx_list_gru.append(idx)

    y_i = pd.DataFrame({"y_test": y_test[j].flatten()}).reset_index()
    idx_table = pd.DataFrame({"idx": idx})
    y_i = pd.merge(y_i, idx_table, how="left", left_on="index", right_on="idx")
    y_i["y_pred"] = np.where(y_i["idx"].isna(), 0, 1)
    preds.append(y_i)

final_preds = pd.concat(preds, 0)

test_y = final_preds["y_test"]
test_pred = final_preds["y_pred"]
# #########################################################


# #########################################################
# 4.1 Backtest model
# #########################################################
cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
acc = accuracy_score(test_y, test_pred)
balanced_acc = balanced_accuracy_score(test_y, test_pred)
f1 = f1_score(test_y, test_pred)

idx, doc_index, pred_summaries = return_df_pred_summaries(Xy_doc_label=test_doc_label, y_pred=test_pred.values,
                                                          df_text=test_article_clean,
                                                          thresh=0.5,
                                                          min_num=1)


df_gold = test_highlight_clean[doc_index]
gold_summaries = [' '.join(df_gold.iloc[j]) for j in range(len(pred_summaries))]
summaries_comp = tuple(zip(pred_summaries, gold_summaries))

scores = rouge_scores(pred_summaries, gold_summaries)

results_dict_gru = {'conf_matrix': cm, 'accuracy': acc, 'balanced_acc': balanced_acc, 'f1': f1,
                'summaries_comp': summaries_comp,
                'sent_index_number': idx, 'Rouge': scores}

pickle.dump(results_dict_gru, open(".//Project/output/GRU_results.pickle", "wb"))
# #########################################################


# #########################################################
# 5.0 LSTM
# #########################################################

# define model
model_lstm = Sequential()
model_lstm.add(LSTM(100, input_shape=(None, 768), return_sequences=True, dropout=0.4))
model_lstm.add(TimeDistributed(Dense(10, activation='relu')))
model_lstm.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model_lstm.compile(loss='binary_crossentropy', optimizer=optimizer,
              metrics=["accuracy"])


# train LSTM
training_loss_avg_lstm, training_metric_val_lstm, training_metric_train_lstm, training_metric_val_loss_lstm = list(), list(), list(), list()
for epoch in range(10):
    for j in range(len(X_train)):
        training_loss = list()
        X, y = X_train[j], y_train[j]
        val_index = randrange(5000)
        X_val, y_val = X_validation[val_index], y_validation[val_index]
        callback_metric = Metrics(model_lstm, train_data=[X, y],val_data=[X_val, y_val])
        history = model_lstm.fit(X, y, validation_data=[X_val, y_val], epochs=1, batch_size=1, callbacks=[callback_metric])
        training_loss.append(history.history['loss'])

        if j % 100 == 0:
            print(f"#### {j} sentences finished!#### ")
    print(f" EPOCH finished! ######## {epoch + 1} EPOCH finished! ########")

    training_loss_avg_lstm.append(np.mean(training_loss))
    training_metric_val_lstm.append(np.mean(callback_metric.get_data()[0]))
    training_metric_train_lstm.append(np.mean(callback_metric.get_data()[1]))
    training_metric_val_loss_lstm.append(np.mean(callback_metric.get_data()[2]))

model_lstm.summary()
model_lstm.save("./Project/interim/model_lstm")

# evaluate LSTM
y_pred_list_lstm, idx_list_lstm, preds = list(), list(), list()
for j in range(len(X_test)):
    X = X_test[j]
    y_pred = model_lstm.predict(X, verbose=0)
    idx = np.argsort(y_pred[0].flatten())[-4:]
    idx = sorted(idx)
    y_pred_list_lstm.append(y_pred)
    idx_list_lstm.append(idx)

    y_i = pd.DataFrame({"y_test": y_test[j].flatten()}).reset_index()
    idx_table = pd.DataFrame({"idx": idx})
    y_i = pd.merge(y_i, idx_table, how="left", left_on="index", right_on="idx")
    y_i["y_pred"] = np.where(y_i["idx"].isna(), 0, 1)
    preds.append(y_i)

final_preds_lstm = pd.concat(preds, 0)

test_y_lstm = final_preds_lstm["y_test"]
test_pred_lstm = final_preds_lstm["y_pred"]
# #########################################################


# #########################################################
# 5.1 Backtest model
# #########################################################
cm_lstm = confusion_matrix(test_y_lstm, test_pred_lstm, labels=[0, 1])
acc_lstm = accuracy_score(test_y_lstm, test_pred_lstm)
balanced_acc_lstm = balanced_accuracy_score(test_y_lstm, test_pred_lstm)
f1_lstm = f1_score(test_y_lstm, test_pred_lstm)

idx_lstm, doc_index, pred_summaries = return_df_pred_summaries(Xy_doc_label=test_doc_label, y_pred=test_pred_lstm.values,
                                                          df_text=test_article_clean,
                                                          thresh=0.5,
                                                          min_num=1)

df_gold = test_highlight_clean[doc_index]
gold_summaries = [' '.join(df_gold.iloc[j]) for j in range(len(pred_summaries))]
summaries_comp_lstm = tuple(zip(pred_summaries, gold_summaries))

scores_lstm = rouge_scores(pred_summaries, gold_summaries)

results_dict_lstm = {'conf_matrix': cm_lstm, 'accuracy': acc_lstm, 'balanced_acc': balanced_acc_lstm, 'f1': f1_lstm,
                    'summaries_comp': summaries_comp_lstm,
                    'sent_index_number': idx_lstm, 'Rouge': scores_lstm}


pickle.dump(results_dict_lstm, open("./Project/output/LSTM_results.pickle", "wb"))
# #########################################################


# #########################################################
# 6.0 Bidirectional LSTM
# #########################################################

# define model
model_bi_lstm = Sequential()
model_bi_lstm.add(Bidirectional(LSTM(100, input_shape=(None, 768), return_sequences=True, dropout=0.4)))
model_bi_lstm.add(TimeDistributed(Dense(10, activation='relu')))
model_bi_lstm.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_bi_lstm.compile(loss='binary_crossentropy', optimizer=optimizer,
              metrics=["accuracy"])


# train Bidirectional LSTM
training_loss_avg_bilstm, training_metric_val_bilstm, training_metric_train_bilstm, training_metric_val_loss_bilstm = list(), list(), list(), list()
for epoch in range(10):
    for j in range(len(X_train)):
        training_loss = list()
        X, y = X_train[j], y_train[j]
        val_index = randrange(5000)
        X_val, y_val = X_validation[val_index], y_validation[val_index]
        callback_metric = Metrics(model_bi_lstm, train_data=[X, y], val_data=[X_val, y_val])
        history = model_bi_lstm.fit(X, y, validation_data=[X_val, y_val], epochs=1, batch_size=1, callbacks=[callback_metric])
        training_loss.append(history.history['loss'])

        if j % 100 == 0:
            print(f"#### {j} sentences finished!#### ")
    print(f" EPOCH finished! ######## {epoch + 1} EPOCH finished! ########")

    training_loss_avg_bilstm.append(np.mean(training_loss))
    training_metric_val_bilstm.append(np.mean(callback_metric.get_data()[0]))
    training_metric_train_bilstm.append(np.mean(callback_metric.get_data()[1]))
    training_metric_val_loss_bilstm.append(np.mean(callback_metric.get_data()[2]))


model_bi_lstm.summary()
model_bi_lstm.save("./Project/interim/model_bi_lstm")

model_bi_lstm = keras.models.load_model("./Project/interim/model_bi_lstm")

result_model_metrics = {"train_loss":training_loss_avg_bilstm, "train_bacc":training_metric_train_bilstm,
                        "val_loss": training_metric_val_loss_bilstm, "val_bacc": training_metric_val_bilstm}

pickle.dump(result_model_metrics, open("./Project/output/BILSTM_train_metric_results.pickle", "wb"))


# evaluate BILSTM
y_pred_list_bilstm, idx_list_bilstm, preds = list(), list(), list()
for j in range(len(X_test)):
    X = X_test[j]
    y_pred = model_bi_lstm.predict(X, verbose=0)
    idx = np.argsort(y_pred[0].flatten())[-4:]
    idx = sorted(idx)
    y_pred_list_bilstm.append(y_pred)
    idx_list_bilstm.append(idx)

    y_i = pd.DataFrame({"y_test": y_test[j].flatten()}).reset_index()
    idx_table = pd.DataFrame({"idx": idx})
    y_i = pd.merge(y_i, idx_table, how="left", left_on="index", right_on="idx")
    y_i["y_pred"] = np.where(y_i["idx"].isna(), 0, 1)
    preds.append(y_i)

final_preds_bilstm = pd.concat(preds, 0)

test_y_bilstm = final_preds_bilstm["y_test"]
test_pred_bilstm = final_preds_bilstm["y_pred"]
# #########################################################


# #########################################################
# 6.1 Backtest model
# #########################################################
cm_bilstm = confusion_matrix(test_y_bilstm, test_pred_bilstm, labels=[0, 1])
acc_bilstm = accuracy_score(test_y_bilstm, test_pred_bilstm)
balanced_acc_bilstm = balanced_accuracy_score(test_y_bilstm, test_pred_bilstm)
f1_bilstm = f1_score(test_y_bilstm, test_pred_bilstm)

idx_bilstm, doc_index, pred_summaries = return_df_pred_summaries(Xy_doc_label=test_doc_label, y_pred=test_pred_bilstm.values,
                                                          df_text=test_article_clean,
                                                          thresh=0.5,
                                                          min_num=1)

df_gold = test_highlight_clean[doc_index]
gold_summaries = [' '.join(df_gold.iloc[j]) for j in range(len(pred_summaries))]
summaries_comp_bilstm = tuple(zip(pred_summaries, gold_summaries))

scores_bilstm = rouge_scores(pred_summaries, gold_summaries)

results_dict_bilstm = {'conf_matrix': cm_bilstm, 'accuracy': acc_bilstm, 'balanced_acc': balanced_acc_bilstm, 'f1': f1_bilstm,
                    'summaries_comp': summaries_comp_bilstm,
                    'sent_index_number': idx_bilstm, 'Rouge': scores_bilstm}


pickle.dump(results_dict_bilstm, open("./Project/output/BILSTM_results.pickle", "wb"))
bilstm = pickle.load(open("./Project/output/BILSTM_results.pickle", "rb"))
# #########################################################


















