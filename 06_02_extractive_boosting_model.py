# #########################################################
# 0.0 Import
# #########################################################
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score

from catboost import CatBoostClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

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

# remove redundant datasets
del train_prep
del test_prep
del validation_prep
# #########################################################


# #########################################################
# 2.0 HyperOpt for CatBoost
# #########################################################
space = {
    'depth': hp.choice("depth", np.arange(4, 8, 1)),
    'min_child_samples': hp.choice("min_child_samples", np.arange(1, 4, 1)),
    'learning_rate': hp.quniform('learning_rate', 0.05, 0.1, 0.001),
    'colsample_bylevel': hp.choice("colsample_bylevel", np.arange(2500, 4000, 1)),
    'leaf_estimation_iterations': hp.choice('leaf_estimation_iterations', np.arange(3, 10, 1)),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 2),
    "bagging_temperature": hp.uniform("bagging_temperature", 0, 0.2)}


def objective(space):
    clf = CatBoostClassifier(depth=int(space["depth"]),
                             min_child_samples=int(space["min_child_samples"]),
                             learning_rate=float(space["learning_rate"]),
                             leaf_estimation_iterations=int(space["leaf_estimation_iterations"]),
                             l2_leaf_reg=float(space["l2_leaf_reg"]),
                             bagging_temperature=float(space["bagging_temperature"]),
                             task_type="GPU",
                             loss_function='Logloss',
                             eval_metric='BalancedAccuracy', auto_class_weights="Balanced", random_state=2022)

    evaluation = [(validation_X, validation_y)]

    clf.fit(train_X, train_y,
            eval_set=evaluation,
            early_stopping_rounds=10, verbose=True)

    pred = clf.predict(test_X)
    acc = balanced_accuracy_score(test_y, pred)
    print("Balanced accuracy score:", acc)
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()

best_hyperparams_cat = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=100,
                            trials=trials)
# #########################################################


# #########################################################
# 3.0 Refit CatBoost
# #########################################################
clf = CatBoostClassifier(depth=int(best_hyperparams_cat["depth"]),
                         min_child_samples=int(best_hyperparams_cat["min_child_samples"]),
                         learning_rate=float(best_hyperparams_cat["learning_rate"]),
                         leaf_estimation_iterations=int(best_hyperparams_cat["leaf_estimation_iterations"]),
                         l2_leaf_reg=float(best_hyperparams_cat["l2_leaf_reg"]),
                         bagging_temperature=float(best_hyperparams_cat["bagging_temperature"]),
                         task_type="GPU",
                         loss_function='Logloss',
                         eval_metric='BalancedAccuracy', auto_class_weights="Balanced", random_state=2022)

evaluation = [(validation_X, validation_y)]

clf.fit(train_X, train_y, verbose=True)

pickle.dump(clf, open("./Project/output/CATBOOST_model.pickle", "wb"))



pred = clf.predict(test_X)
test_prob = clf.predict_proba(test_X)

# #########################################################


# #########################################################
# 4.0 Backtest model
# #########################################################
cm = confusion_matrix(test_y, pred, labels=[0, 1])
acc = accuracy_score(test_y, pred)
balanced_acc = balanced_accuracy_score(test_y, pred)
f1 = f1_score(test_y, pred)

idx, doc_index, pred_summaries = return_df_pred_summaries(Xy_doc_label=test_doc_label, y_pred=pred,
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


pickle.dump(results_dict, open("./Project/output/catboost_results.pickle", "wb"))
# #########################################################


# #########################################################
# 4.1 Backtest model - best 4
# #########################################################
test_prediction_data = pd.DataFrame({"label": test_doc_label.flatten(), "pred": test_prob[:,1]})
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

results_dict_catboost_4 = {'conf_matrix': cm, 'accuracy': acc, 'balanced_acc': balanced_acc, 'f1': f1,
                    'summaries_comp': summaries_comp_4,
                    'sent_index_number': idx_4, 'Rouge': scores}

pickle.dump(results_dict_catboost_4, open("./Project/output/catboost_results_4.pickle", "wb"))
# #########################################################
