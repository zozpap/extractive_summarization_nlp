# #########################################################
# 0.0 Import
# #########################################################
import pickle
import nltk

from Project.functions import preprocess_text

# #########################################################
# 1.0 Preprocessing
# #########################################################
# load interim
train = pickle.load(open("./Project/interim/train.pickle", "rb"))
validation = pickle.load(open("./Project/interim/validation.pickle", "rb"))
test = pickle.load(open("./Project/interim/test.pickle", "rb"))

# load nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
# #########################################################


# #########################################################
# 2.0 Preprocessing - datasets
# #########################################################

# train
train["article" + "_clean"] = train["article"].apply(
    lambda x: preprocess_text(x, punkt=False, lower=False, slang=False, stemm=False, lemm=False, contraction=True,
                              stopwords=False))

train["highlights" + "_clean"] = train["highlights"].apply(
    lambda x: preprocess_text(x, punkt=False, lower=False, slang=False, stemm=False, lemm=False, contraction=True,
                              stopwords=False))

# test
test["article" + "_clean"] = test["article"].apply(
    lambda x: preprocess_text(x, punkt=False, lower=False, slang=False, stemm=False, lemm=False, contraction=True,
                              stopwords=False))

test["highlights" + "_clean"] = test["highlights"].apply(
    lambda x: preprocess_text(x, punkt=False, lower=False, slang=False, stemm=False, lemm=False, contraction=True,
                              stopwords=False))

# validation
validation["article" + "_clean"] = validation["article"].apply(
    lambda x: preprocess_text(x, punkt=False, lower=False, slang=False, stemm=False, lemm=False, contraction=True,
                              stopwords=False))

validation["highlights" + "_clean"] = validation["highlights"].apply(
    lambda x: preprocess_text(x, punkt=False, lower=False, slang=False, stemm=False, lemm=False, contraction=True,
                              stopwords=False))

# Write to pickle
pickle.dump(train, open("./Project/interim/train_cleaned.pickle", "wb"))
pickle.dump(test, open("./Project/interim/test_cleaned.pickle", "wb"))
pickle.dump(validation, open("./Project/interim/validation_cleaned.pickle", "wb"))
