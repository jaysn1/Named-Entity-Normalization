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
    MAX_LEN = 256
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    BATCH_SIZE = 32
    EPOCHS = 20

from reading_data import reading_files, reading_files_test
from create_dataset import create_dataset
from UMLS_methods import *

# load training data
train_data, CUI, iCUI = reading_files("./train")
train_df = create_dataset(train_data)
### removing CUIs from dataset that have only one mentions
train_single_cui = []
for cui, mention in CUI.items():
    if len(mention) == 1:
        train_single_cui.append(cui)

#### Training Phase  ####

# we tokenize the sentences
sentences = train_df.original_sentence
st = train_df.position_start
end = train_df.position_end

tokenizer = config.TOKENIZER
mask_token = tokenizer.tokenize("[MASK]")

tokenized_pre = tokenizer.tokenize(sentences[0][:st[0]])
tokenized_post = tokenizer.tokenize(sentences[0][end[0]:])
tokenized = tokenized_pre + (mask_token) + tokenized_post

ids = tokenizer.convert_tokens_to_ids(tokenized)
target_position = len(tokenized_pre)

tokenizer = config.TOKENIZER
mask_token = tokenizer.tokenize("[MASK]")

input_ids = []
input_target_positions = []
labels = []
enc_label = preprocessing.LabelEncoder()

for index in tqdm(train_df.index):
    row = train_df.loc[index]
    if row['cui'] in train_single_cui:
        continue
    sentence = row['original_sentence']
    st = row['position_start']
    end = row['position_end']

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
labels = enc_label.fit_transform(labels)

import joblib
meta_data = {
    'enc_label': enc_label
}
joblib.dump(meta_data, "meta.bin")

from keras.preprocessing.sequence import pad_sequences
input_ids = pad_sequences(input_ids, maxlen=config.MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

attention_masks = []
for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

from sklearn.model_selection import train_test_split

(train_inputs, validation_inputs, 
 train_position, validation_position, 
    train_labels, validation_labels) = train_test_split(input_ids, input_target_positions, labels, random_state=2020, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=2020, test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_position = torch.tensor(train_position)
validation_position = torch.tensor(validation_position)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels, train_position)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.BATCH_SIZE)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_position)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=config.BATCH_SIZE)


from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = len(enc_label.classes_), # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
# Tell pytorch to run this model on the GPU.
_ = model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
from transformers import get_linear_schedule_with_warmup
epochs = config.EPOCHS
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

import numpy as np
import time
import datetime
import random

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_pos = batch[3].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    ########################################3
    print("")
    print("Running Validation...")
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_pos = batch
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("  Accuracy: {0:.2f}%".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

model.eval()
predictions , true_labels = [], []
prediction_dataloader = train_dataloader
for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels, b_pos = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

print('DONE.')

import torch
torch.save(model, "model.bin")

print("Model and meta files are saved to drive.")