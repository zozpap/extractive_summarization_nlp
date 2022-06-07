# #########################################################
# 0.0 Import
# #########################################################
import pickle
from Project.functions import idx_labels_target, data_info, dataset_creator

# #########################################################


# #########################################################
# 1.0 Load dataset
# #########################################################
train = pickle.load(open("./Project/interim/train_cleaned.pickle", "rb"))
test = pickle.load(open("./Project/interim/test_cleaned.pickle", "rb"))
validation = pickle.load(open("./Project/interim/validation_cleaned.pickle", "rb"))
# #########################################################


# #########################################################
# 2.0 Reset index
# #########################################################
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
validation = validation.reset_index(drop=True)
# #########################################################


# #########################################################
# 3.0 Cosine similarity
# #########################################################
# get index list and target labels
idx_list_train, labels_train = idx_labels_target(train.article_embedding, train.highlights_embedding)
idx_list_test, labels_test = idx_labels_target(test.article_embedding, test.highlights_embedding)
idx_list_validation, labels_validation = idx_labels_target(validation.article_embedding,
                                                           validation.highlights_embedding)

train['labels_idx_list'] = idx_list_train
train['labels'] = labels_train

test['labels_idx_list'] = idx_list_test
test['labels'] = labels_test

validation['labels_idx_list'] = idx_list_validation
validation['labels'] = labels_validation
# #########################################################


# #########################################################
# 4.0 Getting data information (length, mean etc.. )
# #########################################################
train_embed_text, train_doc_label, train_doc_mean, train_doc_length, train_sent_num = data_info(train)

validation_embed_text, validation_doc_label, validation_doc_mean, validation_doc_length, validation_sent_num = data_info(
    validation)

test_embed_text, test_doc_label, test_doc_mean, test_doc_length, test_sent_num = data_info(test)

# labels
train_y_labels = train.labels
test_y_labels = test.labels
validation_y_labels = validation.labels
# #########################################################


# #########################################################
# 5.0 Final dataset creation
# #########################################################
train_prep = dataset_creator(train, train_y_labels, train_doc_label, train_embed_text, train_doc_mean, train_sent_num,
                             train_doc_length)

validation_prep = dataset_creator(validation, validation_y_labels, validation_doc_label, validation_embed_text,
                                  validation_doc_mean, validation_sent_num,
                                  validation_doc_length)

test_prep = dataset_creator(test, test_y_labels, test_doc_label, test_embed_text, test_doc_mean, test_sent_num,
                            test_doc_length)
# #########################################################


# #########################################################
# 6.0 Save output
# #########################################################
pickle.dump(train_prep, open("./Project/interim/train_prep.pickle", "wb"), protocol=4)
pickle.dump(test_prep, open("./Project/interim/test_prep.pickle", "wb"), protocol=4)
pickle.dump(validation_prep, open("./Project/interim/validation_prep.pickle", "wb"), protocol=4)
# #########################################################



