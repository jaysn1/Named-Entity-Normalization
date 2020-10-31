# this notebook only predicts the CUI based on BERT model ie only if there are multiple candidates for a mention.
# Steps to run this notebook.
# 0. Switch to GPU first
# 1. Mount Gdrive with model and meta files using the GUI on left plane.
# 2. Upload the credentials.json file.
# 2. Upload a file "final_df.csv" that will be used by the BERT / use GIT to upload it
# 3. RUN ALL cells
# 4. File called "file_with_prediction.csv" will be generated which can be used to verify\analyse result

# run get_data_for_bert.py => will generate final_df.csv
import get_data_for_bert
get_data_for_bert.main()

import torch, joblib
model = torch.load("model.bin")
meta = joblib.load("meta.bin")
enc_label = meta['enc_label']
le_dict = dict(zip(enc_label.classes_, enc_label.transform(enc_label.classes_)))

import tensorflow as tf
import torch
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn import preprocessing

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
class config:
    TRAIN_PATH = "./train"
    MAX_LEN = 64
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    BATCH_SIZE = 32
    EPOCHS = 10

import numpy as np
import pandas as pd
from tqdm import tqdm

all_df = pd.read_csv("final_df.csv")
test_df = all_df[all_df['prediction_source'] == "BERT"]

for i in test_df.index:
    try:
        test_df.at[i, 'prediction'] = eval(test_df['prediction'].loc[i])
    except:
        print("C")

input_ids = []
input_target_positions = []
labels = []
test_candidates = []
enc_label = enc_label
tokenizer = config.TOKENIZER
mask_token = tokenizer.tokenize("[MASK]")

for index in tqdm(test_df.index):
    row = test_df.loc[index]
    sentence = row['original_sentence']
    st = row['position_start']
    end = row['position_end']
    candidates = row['prediction']

    tokenized_pre = tokenizer.tokenize(sentence[:st])
    target_position = len(tokenized_pre)
    if target_position > config.MAX_LEN//2:
        tokenized_pre = tokenized_pre[-config.MAX_LEN//2:]
        target_position = len(tokenized_pre)

    tokenized_post = tokenizer.tokenize(sentence[end+1:])
    tokenized = tokenized_pre + (mask_token) + tokenized_post
    ids = tokenizer.convert_tokens_to_ids(tokenized)
    ids = ids[(len(ids)-config.MAX_LEN)//2+1 : (len(ids)+config.MAX_LEN)//2-1]
    
    input_ids.append(ids)
    input_target_positions.append(target_position)
    labels.append(row['cui'])
    test_candidates.append([le_dict.get(_, enc_label.transform(['CUI-less'])[0]) for _ in candidates])

labels = [le_dict.get(_, enc_label.transform(['CUI-less'])[0]) for _ in labels] #enc_label.transform(labels)

le_dict.get('C0019699', enc_label.transform(['CUI-less'])[0])

attention_masks = []
for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

max_len = max([len(_) for _ in test_candidates])
for i in range(len(test_candidates)):
    test_candidates[i] = test_candidates[i] + [enc_label.transform(['CUI-less'])[0]] * (max_len - len(test_candidates[i]))
max_len = max([len(_) for _ in test_candidates])

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)
input_target_positions = torch.tensor(input_target_positions)
test_candidates = torch.tensor(test_candidates)

test_data = TensorDataset(input_ids, attention_masks, labels, input_target_positions, test_candidates)
test_sampler = SequentialSampler(test_data)
prediction_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.BATCH_SIZE)

predictions , true_labels = [], []
model.eval()
for batch in tqdm(prediction_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels, b_pos, b_candidates = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    logits = np.argsort(logits, axis=1)
    preds = []
    for i in range(len(logits)):
        for _ in logits[i]:
            if _ in b_candidates[i]:
                pred = _
                break
            else:
                pred = logits[i][0]
        preds.append(pred)
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(preds)
    true_labels.append(label_ids)

from sklearn.metrics import accuracy_score
flat_predictions = [item for sublist in predictions for item in sublist]
flat_true_labels = [item for sublist in true_labels for item in sublist]
bert_acc = accuracy_score(flat_true_labels, flat_predictions)
print("Accuracy: ", bert_acc)

test_df.at[:,'BERT_prediction'] = enc_label.inverse_transform(flat_predictions)

test_df.to_csv("file_with_BERT_prediction.csv")

tdf = all_df[all_df['prediction_source'] != 'BERT']
non_bert_acc = len(tdf[tdf['cui']==tdf['prediction']]) / len(tdf)
print("NON BERT accuracy: ", non_bert_acc)
tdf.to_csv("file_with_non_BERT_prediction.csv")

total_acc = (non_bert_acc * len(tdf) + bert_acc * len(test_df)) / (len(test_df)+len(tdf))
print("total accuracy: ", total_acc)

