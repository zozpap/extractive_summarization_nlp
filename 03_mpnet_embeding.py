# #########################################################
# 0.0 Import
# #########################################################
import pickle
import spacy
# spacy.cli.download("en_core_web_lg")
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from Project.functions import spacy_cleaner, sentence_embedding

tqdm.pandas()
# #########################################################


# #########################################################
# 1.0 Load datasets
# #########################################################
train = pickle.load(open("./Project/interim/train_cleaned.pickle", "rb"))
test = pickle.load(open("./Project/interim/test_cleaned.pickle", "rb"))
validation = pickle.load(open("./Project/interim/validation_cleaned.pickle", "rb"))
# #########################################################


# #########################################################
# 2.0 Spacy pipeline for tok2vec, tagger, parser, senter,
# ner, attribute_ruler, lemmatizer
# #########################################################
nlp = spacy.load("en_core_web_lg")

validation['article_clean_spacy'] = validation['article_clean'].progress_apply(
    lambda text: spacy_cleaner(text, nlp=nlp))
validation['highlights_clean_spacy'] = validation['highlights_clean'].progress_apply(
    lambda text: spacy_cleaner(text, nlp=nlp))

test['article_clean_spacy'] = test['article_clean'].progress_apply(
    lambda text: spacy_cleaner(text, nlp=nlp))
test['highlights_clean_spacy'] = test['highlights_clean'].progress_apply(
    lambda text: spacy_cleaner(text, nlp=nlp))

train['article_clean_spacy'] = train['article_clean'].progress_apply(
    lambda text: spacy_cleaner(text, nlp=nlp))
train['highlights_clean_spacy'] = train['highlights_clean'].progress_apply(
    lambda text: spacy_cleaner(text, nlp=nlp))
# #########################################################


# #########################################################
# 3.0 Embedding
# #########################################################
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cpu")

validation['article_embedding'] = validation['article_clean_spacy'].swifter.apply(
    lambda text: sentence_embedding(text, embedder=embedder))
validation['highlights_embedding'] = validation['highlights_clean_spacy'].swifter.apply(
    lambda text: sentence_embedding(text, embedder=embedder))

test['article_embedding'] = test['article_clean_spacy'].swifter.apply(
    lambda text: sentence_embedding(text, embedder=embedder))
test['highlights_embedding'] = test['highlights_clean_spacy'].swifter.apply(
    lambda text: sentence_embedding(text, embedder=embedder))

train['article_embedding'] = train['article_clean_spacy'].swifter.apply(
    lambda text: sentence_embedding(text, embedder=embedder))
train['highlights_embedding'] = train['highlights_clean_spacy'].swifter.apply(
    lambda text: sentence_embedding(text, embedder=embedder))
# #########################################################


# #########################################################
# 4.0 Save output
# #########################################################
pickle.dump(train, open("./Project/interim/train_cleaned.pickle", "wb"))
pickle.dump(test, open("./Project/interim/test_cleaned.pickle", "wb"))
pickle.dump(validation, open("./Project/interim/validation_cleaned.pickle", "wb"))
# #########################################################
