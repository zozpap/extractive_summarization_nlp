# #########################################################
# 0.0 Import
# #########################################################
from datasets import load_dataset
import pandas as pd
import pickle
import random
# #########################################################


# #########################################################
# 1.0 Load dataset
# #########################################################
data = load_dataset("ccdv/cnn_dailymail", '3.0.0')

train = pd.DataFrame(data["train"])
test = pd.DataFrame(data["test"])
validation = pd.DataFrame(data["validation"])
# #########################################################


# #########################################################
# 2.0 Select subset of each data
# #########################################################
random.seed(2022)

indx = random.sample(range(validation.shape[0]), 5000)
validation = validation.iloc[indx, :]

random.seed(2022)

indx = random.sample(range(test.shape[0]), 5000)
test = test.iloc[indx, :]

random.seed(2022)

indx = random.sample(range(train.shape[0]), 12000)
train = train.iloc[indx, :]
# #########################################################


# #########################################################
# 3.0 Save dataset
# #########################################################
pickle.dump(train, open("./Project/interim/train.pickle", "wb"))
pickle.dump(validation, open("./Project/interim/validation.pickle", "wb"))
pickle.dump(test, open("./Project/interim/test.pickle", "wb"))
# #########################################################

