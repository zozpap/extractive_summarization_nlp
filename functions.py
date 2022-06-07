import re
import nltk
import contractions
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize

from rouge import Rouge

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from collections import Counter

# 02 script
# #########################################################
def preprocess_text(txt, punkt=True, lower=True, slang=True, stemm=False,
                    lemm=True, contraction=True, stopwords=True):
    """
    return clean text format
    :param txt: input text
    :param punkt: punctuation
    :param lower: lower string
    :param slang: slang filter
    :param stemm: stemming
    :param lemm: lemmazation
    :param contraction: contraction replacement
    :param stopwords: stopwords removal
    :return: clean text
    """
    lst_stopwords = list(set(nltk.corpus.stopwords.words("english")))
    lst_stopwords = lst_stopwords + ["cnn", "say", "said", "new", "wa", "ha"]

    contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                           "could've": "could have", "couldn't": "could not", "didn't": "did not",
                           "doesn't": "does not",
                           "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                           "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                           "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                           "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                           "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                           "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                           "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                           "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                           "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                           "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                           "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                           "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                           "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                           "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is",
                           "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                           "they'll've": "they will have", "they're": "they are", "they've": "they have",
                           "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                           "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                           "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                           "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                           "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have",
                           "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                           "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                           "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                           "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                           "you'll've": "you will have", "you're": "you are", "you've": "you have"}

    # separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))

    # remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt

    # strip
    txt = " ".join([word.strip() for word in txt.split()])

    # lowercase
    txt = txt.lower() if lower is True else txt

    # slang
    txt = contractions.fix(txt) if slang is True else txt

    # tokenize (convert from string to list)
    lst_txt = txt.split()

    # contraction mapping
    if contraction:
        for i in range(len(lst_txt)):
            word = lst_txt[i]
            if word in contraction_mapping:
                lst_txt[i] = contraction_mapping[word]

    # stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]

    # lemmazation (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]

    # stopwords
    if stopwords is True:
        lst_txt = [word for word in lst_txt if word not in lst_stopwords]
        lst_txt = ["".join(word.split("cnn")) if "cnn" in word else word for word in lst_txt]

    # back to string
    txt = " ".join(lst_txt)
    return txt


# #########################################################

# 03 script
# #########################################################
def extractive_algo(article, highlight, sentences, method):
    """

    :param article:
    :param highlight:
    :param sentences:
    :param method:
    :return:
    """
    # parser
    parser = PlaintextParser.from_string(article, Tokenizer("english"))

    # extractive summarization method
    textrank_summarizer = method

    # initialize summarizer
    summary = textrank_summarizer(parser.document, sentences)

    # add summarized sentences to a list
    bleu_sentences_lst, rouge_sentences_lst = list(), list()
    for i in range(len(summary)):
        # summary tokenization
        prepared_summary = preprocess_text(summary[i]._text, punkt=False, lower=True, slang=False, stemm=False,
                                           lemm=False, contraction=True, stopwords=False)

        rouge_sentences_lst.append(prepared_summary)

    rouge_highlight_lst = list()
    highlight_splitted = highlight.split(" . ")
    for i in range(sentences):
        # highlight tokenization
        prepared_highlight = preprocess_text(highlight_splitted[i], punkt=False, lower=True, slang=False,
                                             stemm=False, lemm=False, contraction=True, stopwords=False)

        rouge_highlight_lst.append(prepared_highlight)

    # initialize Rouge
    ROUGE = Rouge()

    try:
        rouge = ROUGE._get_scores(rouge_sentences_lst, rouge_highlight_lst)[0]

        rouge_scores = str(rouge["rouge-1"]["r"]) + "," + str(
            rouge["rouge-1"]["p"]) + "," + str(rouge["rouge-1"]["f"]) + "," + str(rouge["rouge-2"]["r"]) + "," + str(
            rouge["rouge-2"]["p"]) + "," + str(rouge["rouge-2"]["f"]) + "," + str(rouge["rouge-l"]["r"]) + "," + str(
            rouge["rouge-l"]["p"]) + "," + str(rouge["rouge-l"]["f"])

    except ValueError:
        rouge_scores = str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0)+ "," + str(0) + "," + str(0) + "," + str(0)

    bleu_gram_mean_and_rouge_scores = rouge_scores

    return str(bleu_gram_mean_and_rouge_scores)


def wordcounter(txt_all,nlp, wordvstoken):

    all_word_counts = Counter()
    for txt in txt_all:
        doc = nlp(txt)

        if wordvstoken:
            # all tokens that arent stop words or punctuations
            words = [token.text
                     for token in doc
                     if not token.is_stop and not token.is_punct]
            word_freq = Counter(words)
            all_word_counts += word_freq
        else:
            # noun tokens that arent stop words or punctuations
            nouns = [token.text
                     for token in doc
                     if (not token.is_stop and
                         not token.is_punct and
                         token.pos_ == "NOUN")]

            word_freq = Counter(nouns)
            all_word_counts += word_freq

    return all_word_counts
# #########################################################


# 04_01 script
# #########################################################
def spacy_cleaner(text,
                  nlp):
    """
    Returns cleaned article sentences
    :param text:
    :param nlp:
    :return:
    """

    # convert to list of sentences
    text = nlp(text)
    sents = list(text.sents)

    # min length sentences
    sents_clean = [sentence.text for sentence in sents if len(sentence) > 3]

    # remove entries with empty list
    sents_clean = [sentence for sentence in sents_clean if len(sentence) != 0]

    return sents_clean


def sentence_embedding(sents_clean, embedder):
    """
    Sentence embedder function
    :param sents_clean: sentences to be embedded
    :param embedder: transformer model
    :return: embed vectorized arrays of sentences
    """

    return np.array(embedder.encode(sents_clean, convert_to_tensor=True))


# #########################################################


# 04_02 script
# #########################################################
def cosine_sim_finder(doc_emedding, summary_sentence_embed):
    """
    "Argmax of the cosine similarity
    :param doc_emedding:
    :param summary_sentence_embed:
    :return: argmax of the cosine similarity matrix
    """
    matrix = cosine_similarity(doc_emedding, summary_sentence_embed)

    return np.argmax(matrix, axis=0)


def idx_labels_target(article, summary):
    """
    Index list of the most similar sentences and their target binary labels
    :param article:
    :param summary:
    :return:
    """
    # initialize zeros for the binary target
    labels = [np.zeros(doc.shape[0]) for doc in article.tolist()]

    # calculate index values for most similar sentences
    idx_list = [np.sort(cosine_sim_finder(article[j], summary[j])) for j
                in range(article.shape[0])]

    for j in range(article.shape[0]):
        labels[j][idx_list[j]] = 1

    return idx_list, labels


def data_info(df):
    """

    :param df: dataset
    :return:
    """
    s_embed_text = df.article_embedding

    # label docs
    s_doc_label = pd.Series(range(df.shape[0]), name='doc_label')

    # calculate doc mean
    s_doc_mean = s_embed_text.apply(lambda x: x.mean(axis=0).reshape(1, -1))

    # calculate doc sent length
    s_doc_length = s_embed_text.apply(lambda x: x.shape[0])

    # create values for each sentence in doc
    X_doc_label_list, X_doc_mean_list, X_doc_length_list, X_sent_num_list = list(), list(), list(), list()

    for i in range(len(df)):
        X_doc_mean = s_doc_mean[i]
        X_doc_length = s_doc_length[i]

        X_doc_label_list.append(np.vstack([s_doc_label[i]] * X_doc_length))
        X_doc_mean_list.append(np.vstack([X_doc_mean] * X_doc_length))
        X_doc_length_list.append(np.vstack([X_doc_length] * X_doc_length))
        X_sent_num_list.append(np.array(list(range(X_doc_length))).reshape(-1, 1))

    return s_embed_text, pd.Series(X_doc_label_list), pd.Series(X_doc_mean_list), pd.Series(
        X_doc_length_list), pd.Series(X_sent_num_list)


def dataset_creator(df, y_labels, doc_label, embed_text, doc_mean, sent_num, doc_length):
    """

    :param df:
    :param y_labels:
    :param doc_label:
    :param embed_text:
    :param doc_mean:
    :param sent_num:
    :param doc_length:
    :return:
    """

    # recursive population
    f = np.vectorize(lambda x: x if type(x) == np.ndarray else np.array([[x]]))
    X_lst, y_lst, Xy_doc_label_lst = list(), list(), list()
    for j in range(0, len(df)):
        Xy_doc_label_new = doc_label.values[j]

        X_text_new = embed_text[j]
        X_sent_num_new = sent_num[j]
        X_doc_mean_new = doc_mean[j]
        X_doc_length_new = f(doc_length[j])
        y_new = y_labels[j].reshape(-1, 1)

        X_new = np.hstack((X_text_new, X_doc_mean_new, X_sent_num_new, X_doc_length_new))

        X_lst.append(X_new)
        y_lst.append(y_new)
        Xy_doc_label_lst.append(Xy_doc_label_new)

    X = np.concatenate(X_lst, 0)
    y = np.concatenate(y_lst, 0)
    Xy_doc_label = np.concatenate(Xy_doc_label_lst, 0)

    del X_lst, y_lst, Xy_doc_label_lst
    # wrap X in dataframe with lables
    labels_text_embedding = ['sent_embed_' + str(j) for j in range(768)]
    labels_doc_mean = ['doc_embed_' + str(j) for j in range(768)]
    other_labels = ['sent_number', 'doc_Length']
    col_names = labels_text_embedding + labels_doc_mean + other_labels

    df_X = pd.DataFrame(X, columns=col_names)

    return {'df_original': df, 'Xy_doc_label_array': Xy_doc_label,
            'df_X': df_X, 'y_array': y}


# 04_03 script
# #########################################################
def return_greater_than_min_num(arr, thresh=0.5, min_num=1):
    idx_prelim = np.where(arr >= thresh)

    if idx_prelim[0].shape[0] <= min_num:
        idx = np.argsort(arr)[-min_num:]
        return idx
    else:
        idx = idx_prelim
        return sorted(idx)


def return_df_pred_summaries(Xy_doc_label, y_pred, df_text, thresh, min_num):
    df_label_pred = pd.DataFrame({'doc_label': Xy_doc_label.flatten(),
                                  'y_pred': y_pred.flatten()})

    df_label_pred = df_label_pred.groupby('doc_label').agg(list)

    df_label_pred = df_label_pred.applymap(lambda x: np.array(x))

    df_label_pred = df_label_pred.applymap(lambda arr: return_greater_than_min_num(arr, thresh=thresh,
                                                                                   min_num=min_num))

    # Return predicted summary
    df_doc = df_text[df_label_pred.index]

    pred_summaries = [np.array(df_doc.iloc[j])[df_label_pred.iloc[j][0]].tolist()
                      for j in range(len(df_label_pred))]

    pred_summaries = [summ_list if type(summ_list) == str else
                      ' '.join(summ_list) for summ_list in pred_summaries]

    return df_label_pred.values, df_label_pred.index, pred_summaries


def rouge_scores(pred_summaries, gold_summaries):

    # Calculate rouge scores
    ROUGE = Rouge()

    return ROUGE._get_avg_scores(pred_summaries, gold_summaries)
# #########################################################

