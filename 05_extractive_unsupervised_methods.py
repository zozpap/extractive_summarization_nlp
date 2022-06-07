# #########################################################
# 0.0 Import
# #########################################################
import numpy as np
import pickle
import pandas as pd
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer

from Project.functions import extractive_algo

# #########################################################


# #########################################################
# 1.0 Load dataset
# #########################################################
train = pickle.load(open("./Project/interim/train_cleaned.pickle", "rb"))
test = pickle.load(open("./Project/interim/test_cleaned.pickle", "rb"))
validation = pickle.load(open("./Project/interim/validation_cleaned.pickle", "rb"))

# backtest on test sample
test["sentences"] = test["highlights_clean"].apply(lambda x: len(x.split(" .\n")), 0)
test_data_extractive = test[["article_clean", "highlights_clean", "sentences"]]
# #########################################################


# #########################################################
# 2.0 Extractive algorithm vectorized
# #########################################################
extractive_algo_vectorized = np.vectorize(extractive_algo)
# #########################################################


# #########################################################
# 2.1 Text ranking
# #########################################################
test_data_extractive["textrank"] = extractive_algo_vectorized(test_data_extractive.article_clean,
                                                              test_data_extractive.highlights_clean,
                                                              test_data_extractive.sentences,
                                                              TextRankSummarizer())
test_data_extractive[
    ["Textrank_ROUGE_1_recall", "Textrank_ROUGE_1_precision", "Textrank_ROUGE_1_fscore",
     "Textrank_ROUGE_2_recall", "Textrank_ROUGE_2_precision", "Textrank_ROUGE_2_fscore", "Textrank_ROUGE_L_recall",
     "Textrank_ROUGE_L_precision", "Textrank_ROUGE_L_fscore"]] = test_data_extractive.textrank.str.split(',',
                                                                                                         expand=True)
test_data_extractive = test_data_extractive.drop("textrank", 1)

test_data_extractive[["Textrank_ROUGE_1_recall", "Textrank_ROUGE_1_precision", "Textrank_ROUGE_1_fscore",
                      "Textrank_ROUGE_2_recall", "Textrank_ROUGE_2_precision", "Textrank_ROUGE_2_fscore",
                      "Textrank_ROUGE_L_recall",
                      "Textrank_ROUGE_L_precision", "Textrank_ROUGE_L_fscore"]] = test_data_extractive[
    ["Textrank_ROUGE_1_recall", "Textrank_ROUGE_1_precision", "Textrank_ROUGE_1_fscore",
     "Textrank_ROUGE_2_recall", "Textrank_ROUGE_2_precision", "Textrank_ROUGE_2_fscore", "Textrank_ROUGE_L_recall",
     "Textrank_ROUGE_L_precision", "Textrank_ROUGE_L_fscore"]].apply(pd.to_numeric)
# #########################################################


# #########################################################
# 2.2 Lex ranking
# #########################################################
test_data_extractive["lexrank"] = extractive_algo_vectorized(test_data_extractive.article_clean,
                                                             test_data_extractive.highlights_clean,
                                                             test_data_extractive.sentences,
                                                             LexRankSummarizer())
test_data_extractive[
    ["Lexrank_ROUGE_1_recall", "Lexrank_ROUGE_1_precision", "Lexrank_ROUGE_1_fscore",
     "Lexrank_ROUGE_2_recall", "Lexrank_ROUGE_2_precision", "Lexrank_ROUGE_2_fscore", "Lexrank_ROUGE_L_recall",
     "Lexrank_ROUGE_L_precision", "Lexrank_ROUGE_L_fscore"]] = test_data_extractive.lexrank.str.split(',', expand=True)
test_data_extractive = test_data_extractive.drop("lexrank", 1)

test_data_extractive[["Lexrank_ROUGE_1_recall", "Lexrank_ROUGE_1_precision", "Lexrank_ROUGE_1_fscore",
                      "Lexrank_ROUGE_2_recall", "Lexrank_ROUGE_2_precision", "Lexrank_ROUGE_2_fscore",
                      "Lexrank_ROUGE_L_recall",
                      "Lexrank_ROUGE_L_precision", "Lexrank_ROUGE_L_fscore"]] = test_data_extractive[
    ["Lexrank_ROUGE_1_recall", "Lexrank_ROUGE_1_precision", "Lexrank_ROUGE_1_fscore",
     "Lexrank_ROUGE_2_recall", "Lexrank_ROUGE_2_precision", "Lexrank_ROUGE_2_fscore", "Lexrank_ROUGE_L_recall",
     "Lexrank_ROUGE_L_precision", "Lexrank_ROUGE_L_fscore"]].apply(pd.to_numeric)
# #########################################################


# #########################################################
# 2.3 Luhn ranking
# #########################################################
test_data_extractive["luhnrank"] = extractive_algo_vectorized(test_data_extractive.article_clean,
                                                              test_data_extractive.highlights_clean,
                                                              test_data_extractive.sentences,
                                                              LuhnSummarizer())
test_data_extractive[
    ["Luhnrank_ROUGE_1_recall", "Luhnrank_ROUGE_1_precision", "Luhnrank_ROUGE_1_fscore",
     "Luhnrank_ROUGE_2_recall", "Luhnrank_ROUGE_2_precision", "Luhnrank_ROUGE_2_fscore", "Luhnrank_ROUGE_L_recall",
     "Luhnrank_ROUGE_L_precision", "Luhnrank_ROUGE_L_fscore"]] = test_data_extractive.luhnrank.str.split(',',
                                                                                                         expand=True)
test_data_extractive = test_data_extractive.drop("luhnrank", 1)

test_data_extractive[["Luhnrank_ROUGE_1_recall", "Luhnrank_ROUGE_1_precision", "Luhnrank_ROUGE_1_fscore",
                      "Luhnrank_ROUGE_2_recall", "Luhnrank_ROUGE_2_precision", "Luhnrank_ROUGE_2_fscore",
                      "Luhnrank_ROUGE_L_recall",
                      "Luhnrank_ROUGE_L_precision", "Luhnrank_ROUGE_L_fscore"]] = test_data_extractive[
    ["Luhnrank_ROUGE_1_recall", "Luhnrank_ROUGE_1_precision", "Luhnrank_ROUGE_1_fscore",
     "Luhnrank_ROUGE_2_recall", "Luhnrank_ROUGE_2_precision", "Luhnrank_ROUGE_2_fscore", "Luhnrank_ROUGE_L_recall",
     "Luhnrank_ROUGE_L_precision", "Luhnrank_ROUGE_L_fscore"]].apply(pd.to_numeric)
# #########################################################


# #########################################################
# 2.4 LSA ranking
# #########################################################
test_data_extractive["lsarank"] = extractive_algo_vectorized(test_data_extractive.article_clean,
                                                             test_data_extractive.highlights_clean,
                                                             test_data_extractive.sentences,
                                                             LsaSummarizer())
test_data_extractive[
    ["Lsarank_ROUGE_1_recall", "Lsarank_ROUGE_1_precision", "Lsarank_ROUGE_1_fscore",
     "Lsarank_ROUGE_2_recall", "Lsarank_ROUGE_2_precision", "Lsarank_ROUGE_2_fscore", "Lsarank_ROUGE_L_recall",
     "Lsarank_ROUGE_L_precision", "Lsarank_ROUGE_L_fscore"]] = test_data_extractive.lsarank.str.split(',',
                                                                                                      expand=True)
test_data_extractive = test_data_extractive.drop("lsarank", 1)

test_data_extractive[["Lsarank_ROUGE_1_recall", "Lsarank_ROUGE_1_precision", "Lsarank_ROUGE_1_fscore",
                      "Lsarank_ROUGE_2_recall", "Lsarank_ROUGE_2_precision", "Lsarank_ROUGE_2_fscore",
                      "Lsarank_ROUGE_L_recall",
                      "Lsarank_ROUGE_L_precision", "Lsarank_ROUGE_L_fscore"]] = test_data_extractive[
    ["Lsarank_ROUGE_1_recall", "Lsarank_ROUGE_1_precision", "Lsarank_ROUGE_1_fscore",
     "Lsarank_ROUGE_2_recall", "Lsarank_ROUGE_2_precision", "Lsarank_ROUGE_2_fscore", "Lsarank_ROUGE_L_recall",
     "Lsarank_ROUGE_L_precision", "Lsarank_ROUGE_L_fscore"]].apply(pd.to_numeric)
# #########################################################


# #########################################################
# 2.5 KL ranking
# #########################################################
test_data_extractive["klrank"] = extractive_algo_vectorized(test_data_extractive.article_clean,
                                                            test_data_extractive.highlights_clean,
                                                            test_data_extractive.sentences,
                                                            KLSummarizer())
test_data_extractive[
    ["KLrank_ROUGE_1_recall", "KLrank_ROUGE_1_precision", "KLrank_ROUGE_1_fscore",
     "KLrank_ROUGE_2_recall", "KLrank_ROUGE_2_precision", "KLrank_ROUGE_2_fscore", "KLrank_ROUGE_L_recall",
     "KLrank_ROUGE_L_precision", "KLrank_ROUGE_L_fscore"]] = test_data_extractive.klrank.str.split(',',
                                                                                                   expand=True)
test_data_extractive = test_data_extractive.drop("klrank", 1)

test_data_extractive[["KLrank_ROUGE_1_recall", "KLrank_ROUGE_1_precision", "KLrank_ROUGE_1_fscore",
                      "KLrank_ROUGE_2_recall", "KLrank_ROUGE_2_precision", "KLrank_ROUGE_2_fscore",
                      "KLrank_ROUGE_L_recall",
                      "KLrank_ROUGE_L_precision", "KLrank_ROUGE_L_fscore"]] = test_data_extractive[
    ["KLrank_ROUGE_1_recall", "KLrank_ROUGE_1_precision", "KLrank_ROUGE_1_fscore",
     "KLrank_ROUGE_2_recall", "KLrank_ROUGE_2_precision", "KLrank_ROUGE_2_fscore", "KLrank_ROUGE_L_recall",
     "KLrank_ROUGE_L_precision", "KLrank_ROUGE_L_fscore"]].apply(pd.to_numeric)
# #########################################################


# #########################################################
# 3.0 Save results
# #########################################################

finish = test_data_extractive[["Textrank_ROUGE_1_recall", "Textrank_ROUGE_1_precision", "Textrank_ROUGE_1_fscore",
                               "Textrank_ROUGE_2_recall", "Textrank_ROUGE_2_precision", "Textrank_ROUGE_2_fscore",
                               "Textrank_ROUGE_L_recall",
                               "Textrank_ROUGE_L_precision", "Textrank_ROUGE_L_fscore",
                               "Lexrank_ROUGE_1_recall", "Lexrank_ROUGE_1_precision", "Lexrank_ROUGE_1_fscore",
                               "Lexrank_ROUGE_2_recall", "Lexrank_ROUGE_2_precision", "Lexrank_ROUGE_2_fscore",
                               "Lexrank_ROUGE_L_recall",
                               "Lexrank_ROUGE_L_precision", "Lexrank_ROUGE_L_fscore",
                               "Luhnrank_ROUGE_1_recall", "Luhnrank_ROUGE_1_precision", "Luhnrank_ROUGE_1_fscore",
                               "Luhnrank_ROUGE_2_recall", "Luhnrank_ROUGE_2_precision", "Luhnrank_ROUGE_2_fscore",
                               "Luhnrank_ROUGE_L_recall",
                               "Luhnrank_ROUGE_L_precision", "Luhnrank_ROUGE_L_fscore",
                               "Lsarank_ROUGE_1_recall", "Lsarank_ROUGE_1_precision", "Lsarank_ROUGE_1_fscore",
                               "Lsarank_ROUGE_2_recall", "Lsarank_ROUGE_2_precision", "Lsarank_ROUGE_2_fscore",
                               "Lsarank_ROUGE_L_recall",
                               "Lsarank_ROUGE_L_precision", "Lsarank_ROUGE_L_fscore",
                               "KLrank_ROUGE_1_recall", "KLrank_ROUGE_1_precision", "KLrank_ROUGE_1_fscore",
                               "KLrank_ROUGE_2_recall", "KLrank_ROUGE_2_precision", "KLrank_ROUGE_2_fscore",
                               "KLrank_ROUGE_L_recall",
                               "KLrank_ROUGE_L_precision", "KLrank_ROUGE_L_fscore"
                               ]].mean(axis=0)

pickle.dump(finish, open(".//Project/output/test_data_extractive_simple_results_2.pickle", "wb"))
