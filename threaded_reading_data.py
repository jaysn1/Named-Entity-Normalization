# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:51:36 2020

@author: Jaysn
"""
import os
from collections import defaultdict
import UMLS_methods
import concurrent.futures
import pandas as pd

# specific for training data
def reading_files(base_directory = "train"):
    """
    This method takes a base directory as input and 
    maps the norm fles and notes to a array and dictionary.
    return 
        1. data: array of dictionary(dict contains the file name, norm and note)
        2. CUI dictionary contains text and index for the dictionary in data
        3. iCUI dictionary with inverse mapping from mention to CUIs
    """

    # reading train_file_list.txt to get list of training notes files
    files = []
    with open("{}/train_file_list.txt".format(base_directory)) as f:
        for line in f.readlines():
            files.append(line.strip())
    print("Total number of files: ", len(files))

    # reading list of CUIs
    CUI = {}
    with open("{}/train_norm.txt".format(base_directory)) as f:
        for line in f.readlines():
            CUI[line.strip()] = set()

    print("Total CUIs: ", len(CUI))

    # reading norm and note files from list of files
    data, iCUI = [], defaultdict(set)
    for _, filename in enumerate(files):
        norm_filename = "{}/train_norm/{}.norm".format(base_directory, filename)
        note_filename = "{}/train_note/{}.txt".format(base_directory, filename)

        with open(norm_filename) as f1:
            with open(note_filename) as f2:
                data.append({'name':filename, 
                             'norm':list(map(lambda x: x.strip().split("||"), f1.readlines())), 
                             'note':" ".join(list(map(lambda x: x.strip(), f2.readlines())))    })
        for x in data[-1]['norm']:
            if len(x)>=4:
                i,cui = x[0], x[1]
                mentions = []
                for i in range(2, len(x), 2):
                    mention = (data[-1]['note'][int(x[i]):int(x[i+1])]).strip()
                    mentions.append(mention)
                    iCUI[mention].add(cui)
                CUI[cui].add("|".join(mentions))
            else:
                raise ValueError("{} is wrong".format(x))

    print("Total Data: ", len(data))

    return data, CUI, iCUI

# if __name__ == "__main__":
base_dir = "C:/Users/Jaysn/Anaconda3/envs/NLP/Clinical-Entity-Normalization/train"
data, CUI, iCUI = reading_files(base_dir)

import pandas as pd
from tqdm import tqdm
def create_dataset(data):
    """
    This method will convert the text/notes to dataset by replacing one by one the mention by [MASK],
    thus creating gigantic amount of data.

    input:- data: list of dictionary (note, norm, filename)
    output:- pandas dataframe with original sentence, masked sentence, mention, CUI 
    """
    dataset = []
    sentence_pad = 200
    a = 0
    for d in data:
        note = d['note']
        norm = d['norm']
        
        for x in tqdm(norm):
            a+= 1
            _, cui = x[0], x[1]
            for i in range(2, len(x), 2):
                st, end = int(x[i]), int(x[i+1])
                if st >= sentence_pad and len(note) - end >=sentence_pad:
                    st_ind, end_ind = st - sentence_pad, end + sentence_pad
                elif st >= sentence_pad:
                    st_ind, end_ind = st - sentence_pad, len(note)
                elif len(note) - end >= sentence_pad:
                    st_ind, end_ind = 0, end + sentence_pad
                else:
                    st_ind, end_ind = 0, len(note)
                
                # while st_ch != " ":
                #     st -= 1
                # while end_ch != " ":
                #     end += 1
                
                answers = UMLS_methods.find_mention_in_UMLS_partial_name(note[st:end+1])[:10]
                answer_from_passage=""
                answer_passage = " , ".join([answer['name'] for answer in answers])
                ch_count = 0
                CUIs = ",".join([answer['cui'] for answer in answers])
                for answer in answers:
                    if answer['cui'] == cui:
                        answer_from_passage = answer['name']
                        break
                    ch_count += len(answer['name']) + 3
                
                dataset.append({
                    'cui': cui,
                    # 'original_sentence': note[st_ind:end_ind+1],
                    'BertQAInput': "What does " + note[st:end+1] + " mean in: " + note[st_ind:end_ind] + " ?",
                    # 'BertQAInput': note[st:end+1],
                    # 'masked_sentence': note[:st] + "[MASK]" + note[end+1:],
                    'mention': note[st:end+1],
                    # 'questionLength': len(("What matches the concept in: " + note[st_ind:st] + " <concept> " + note[st:end+1] + " </concept> " + note[end+1:end_ind] + " ?").split(" ")),
                    'answer_passage': answer_passage,
                    'answer': answer_from_passage,
                    'answer_start': ch_count,
                    'answer_end': ch_count + len(answer_from_passage),
                    'CUIs': CUIs
                    # 'position_start': st,
                    # 'position_end': end
                    }
                )
            # break
        # if a >= 100:
        # break
            # break
        # break
    return pd.DataFrame(dataset)


with concurrent.futures.ThreadPoolExecutor() as executor:
    future = [
        executor.submit(create_dataset, data[i*5:(i+1)*5])
        for i in range(10)]

result=pd.DataFrame()
for f in future:
    result=pd.concat([result, f.result()], axis=0)

result.to_csv("final_df.csv")

# dataset = create_dataset(data)
# dataset.to_csv("final_df.csv")
    # cui = "C0333307"
    # print("For CUI: ", cui)
    # print(CUI[cui])
    # for _, mention in CUI[cui]:
    #     print(mention, " : ", data[_]['note'][:100], end="\n\n")

    # test_base_dir = "C:/Users/monil/Desktop/BMI 598 - NLP/Project/Clinical-Entity-Normalization/testing"
    # test_data, _ = reading_files_test(test_base_dir)
    # print(test_data[0])
