# #########################################################
# 0.0 Import
# #########################################################
import seaborn as sb
import pickle
import pandas as pd
from scipy import stats as s
import numpy
from matplotlib import pyplot as plt
from tqdm import tqdm
import spacy
from collections import Counter
from Project.functions import return_df_pred_summaries, rouge_scores, wordcounter
import numpy as np
import statsmodels.api as sm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# #########################################################
# 1.0 Load
# #########################################################
# load interim
train = pickle.load(open("./Project/interim/train_cleaned.pickle", "rb"))
validation = pickle.load(open("./Project/interim/validation_cleaned.pickle", "rb"))
test = pickle.load(open("./Project/interim/test_cleaned.pickle", "rb"))
# #########################################################


# #########################################################
# 2.0 Preprocessing
# #########################################################
sent_length = pd.DataFrame(train.article_clean_spacy.apply(len).values, columns=['Article'])
sent_length['Summary'] = train.highlights_clean_spacy.apply(len).values
result = pd.DataFrame(sent_length.mean().apply(lambda x: round(x, 3)), columns=['mean'])
result["mode"] = s.mode(sent_length.Article.tolist())[0][0], s.mode(sent_length.Summary.tolist())[0][0]
result["median"] = sent_length.Article.median(), sent_length.Summary.median()
result["std"] = sent_length.Article.std(), sent_length.Summary.std()
result["min"] = sent_length.Article.min(), sent_length.Summary.min()
result["max"] = sent_length.Article.max(), sent_length.Summary.max()

# #########################################################
# 3.0 EDA
# #########################################################


# #########################################################
# 3.1 statistics of lengths for each
# #########################################################
sent_length_quant = sent_length.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
sent_length_quant.index.name = 'Percentile'
sent_length_quant.columns.name = 'Number of Sentences'

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
axes[1].hist(sent_length.Summary.values, numpy.linspace(0, 20, 20), label='Summary', color="grey", lw=0)
axes[0].hist(sent_length.Article.values, numpy.linspace(0, 270, 40), label='Articles', color="0", lw=0)
fig.tight_layout()
fig.legend(loc='upper right')
plt.show()


# #########################################################
# 3.2 word counting
# #########################################################
spacy.cli.download("en_core_web_lg")

nlp = spacy.load('en_core_web_lg')

tqdm.pandas()

word_count_train_article = train["article_clean_spacy"].progress_apply(lambda x: wordcounter(x, nlp, True))
word_count_train_sum = train["highlights_clean_spacy"].progress_apply(lambda x: wordcounter(x, nlp, True))

word_count_test_article = test["article_clean_spacy"].progress_apply(lambda x: wordcounter(x, nlp, True))
word_count_test_sum = test["highlights_clean_spacy"].progress_apply(lambda x: wordcounter(x, nlp, True))

word_count_validation_article = validation["article_clean_spacy"].progress_apply(lambda x: wordcounter(x, nlp, True))
word_count_validation_sum = validation["highlights_clean_spacy"].progress_apply(lambda x: wordcounter(x, nlp, True))


def counter(data):
    result = Counter()
    for i in data:
        result += i
    return result


train_article_words = counter(word_count_train_article)
test_article_words = counter(word_count_test_article)
validation_article_words = counter(word_count_validation_article)

train_summary_words = counter(word_count_train_sum)
test_summary_words = counter(word_count_test_sum)
validation_summary_words = counter(word_count_validation_sum)

word_count_article = dict(
    Counter(train_article_words) + Counter(test_article_words) + Counter(validation_article_words))
most_freq_20_words_article = dict(sorted(word_count_article.items(), key=lambda item: item[1], reverse=True))

word_count_summary = dict(
    Counter(train_summary_words) + Counter(test_summary_words) + Counter(validation_summary_words))
most_freq_20_words_summary = dict(sorted(word_count_summary.items(), key=lambda item: item[1], reverse=True))

data_article = pd.DataFrame(list(most_freq_20_words_article.items())[:22],
                            columns=["Article frequent words", "Article frequent words"])
data_sum = pd.DataFrame(list(most_freq_20_words_summary.items())[:22],
                        columns=["Summary frequent words", "Summary frequent words"])

finish_freq_words = pd.concat([data_article, data_sum], 1)
finish_freq_words.to_csv("freq_20_words.csv")


# #########################################################
# 3.3 regression
# #########################################################
train['summary_len'] = train.highlights_clean_spacy.apply(len).values
train['article_len'] = train.article_clean_spacy.apply(len).values

test['summary_len'] = test.highlights_clean_spacy.apply(len).values
test['article_len'] = test.article_clean_spacy.apply(len).values

validation['summary_len'] = validation.highlights_clean_spacy.apply(len).values
validation['article_len'] = validation.article_clean_spacy.apply(len).values

sum_len = pd.concat([train['summary_len'], test['summary_len'], validation['summary_len']], 0)
article_len = pd.concat([train['article_len'], test['article_len'], validation['article_len']], 0)

full_regression = pd.concat([article_len, sum_len], 1).reset_index()

mod = sm.OLS(full_regression["article_len"], full_regression["summary_len"])

res = mod.fit()

with open('summary.csv', 'w') as fh:
    fh.write(res.summary().as_csv())

# create a plot
ypred = res.predict(full_regression["summary_len"])
sb.scatterplot(x=full_regression["summary_len"].values, y=full_regression["article_len"], color='grey',
               label="Dataset points")
sb.lineplot(x=full_regression["summary_len"].values, y=ypred, color='black', label="Least square regression line")
plt.xlabel('Summary length')
plt.ylabel('Article length')
plt.title("Dataset with prediction")
plt.show()


# #########################################################
# 3.4 reference scores
# #########################################################

# best 4 sentences vs higlight rouge score
test = test.reset_index(drop=True)
test["gold_4"] = test["highlights_clean_spacy"].apply(lambda x: x[0:3])

article = [' '.join(test["article_clean_spacy"].iloc[j]) for j in range(len(test["article_clean_spacy"]))]
gold_summaries_4 = [' '.join(test["gold_4"].iloc[j]) for j in range(len(test["gold_4"]))]

scores_best4 = rouge_scores(gold_summaries_4, article)


# cosine similarity sentences vs higlight rouge score
test = pickle.load(open("./Project/interim/test_cleaned.pickle", "rb"))
test_prep = pickle.load(open("./Project/interim/test_prep.pickle", "rb"))

test_data = test_prep["df_original"]
gold_highlight = [' '.join(test["highlights_clean_spacy"].iloc[j]) for j in range(len(test["highlights_clean_spacy"]))]

cos_sentences = [
    ' '.join(list(np.array(test_data["article_clean_spacy"].iloc[j])[test_data["labels_idx_list"].iloc[j]])) for j in
    range(len(test_data["article_clean_spacy"]))]

scores_orig = rouge_scores(cos_sentences, gold_highlight)
