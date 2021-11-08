#!/usr/bin/env python
# coding: utf-8

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import sample
from tqdm import tqdm
import pandas as pd
import time


class TextCrumble(Dataset):

    def __init__(self,tokenizer,n=None,window=254):

        self.nbreaks = n
        self.tokenizer = tokenizer
        self.paragraphs = []
        self.breaks_pos = []
        self.breaks_neg = []
        self.breakpoints = []
        self.tokenized = []
        self.cls, self.sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        self.window = window

    def __len__(self):
        #return len(self.paragraphs)
        return len(self.breakpoints)

    def __getitem__(self,index):

        label = self.breakpoints[index] in self.breaks_pos
        left,right = self.get_pair_even(self.breakpoints[index])
        return self.pair_to_tensor(self.breakpoints[index],left,right,label)

    def read(self,textfile):

        l = open(textfile,"r").readlines()
        l = [s for s in l if not ((len(s.strip()) == 0) or ("THE END" in s))]
        negatives = []

        for i in range(len(l)):
            sent = l[i]
            if "BREAK" in sent:
                self.breaks_pos.append(len(self.paragraphs))
            else:
                t = self.tokenizer.tokenize(sent)
                self.tokenized.append(t)
                self.paragraphs.append(self.tokenizer.convert_tokens_to_ids(t))

        self.breaks_neg = [i for i in range(len(self.paragraphs)) if not i in self.breaks_pos]
        if self.nbreaks:
            self.breaks_neg = sample(self.breaks_neg,self.nbreaks-len(self.breaks_pos))
        #self.breakpoints = self.breaks_pos + self.breaks_neg
        self.breakpoints = range(0,len(self.paragraphs))

    def get_pair_even(self,index):

        left_index = index-1
        left = []

        while (left_index > -1) and (len(left) < self.window):

            this_left = self.paragraphs[left_index]
            len_trunc = self.window-len(left)

            if len(this_left) > len_trunc:
                this_left = this_left[-len_trunc:]

            left = this_left + left
            left_index -= 1

        right_index = index
        right = []

        while (right_index < len(self.paragraphs)) and (len(right) < self.window):

            this_right = self.paragraphs[right_index]
            len_trunc = self.window-len(right)

            if len(this_right) > len_trunc:
                this_right = this_right[:len_trunc]

            right = right + this_right
            right_index += 1

        if len(left) < self.window:
            left = [0]*(self.window-len(left)) + left
        if len(right) < self.window:
            right = right + [0]*(self.window-len(right))

        return(left,right)

    def pair_to_tensor(self,itemindex,toks1,toks2,itemlabel):

        ids1 = [self.cls] + toks1 + [self.sep]
        ids2 = toks2 + [self.sep]

        indexed_tokens = ids1 + ids2
        segments_ids = [0] * len(ids1) + [1] * len(ids2)
        attention_masks = [1] * len(indexed_tokens)

        tokens_tensor = torch.tensor(indexed_tokens)#.view(1,-1)
        segments_tensors = torch.tensor(segments_ids)#.view(1,-1)
        attention_tensor = torch.tensor(attention_masks)#.view(1,-1)

        #print(len(indexed_tokens),len(segments_ids),len(attention_masks))
        #return indexed_tokens,segments_ids#,attention_masks
        return torch.tensor(itemindex),tokens_tensor,segments_tensors,attention_tensor,torch.tensor(itemlabel)



def process_crumble(crumble,textloader,bert,device="cpu"):

    predicted_breaks = []
    for (ilist,t,s,a,llist) in textloader:
    #print("final",len(t),len(s))#,len(a))
    #print(t.shape,s.shape)
    #model.eval()
        t = t.to(device)
        s = s.to(device)
        a = a.to(device)
        pred = bert(t, token_type_ids=s, attention_mask=a)
        predlabel = np.array(torch.argmax(pred.logits,dim=1).cpu())
        softmax = torch.nn.Softmax(dim=1)
        predprob = softmax(pred[0]).detach().cpu().numpy()
        #pb = list(np.where(pred == 1)[0])

        pb_batch = list(np.array(ilist))
        pb_label = list(np.array(llist))

        predicted_breaks += [(pb_batch[i],pb_label[i],predlabel[i],predprob[i],crumble.tokenized[pb_batch[i]]) for i in range(len(pb_batch))]
        #break

    return predicted_breaks


if __name__ == '__main__':

    batch_i = 0
    breaks_table = []
    #nbreaks = 200
    nbreaks = None

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    model = transformers.BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    model.eval()

    #for fi in range(1,301):
    for fi in tqdm(range(1,3)):
        t1 = TextCrumble(tokenizer,n=nbreaks)
        fname = "../santanlp-corpus/corpus1/test/{0:0=3d}.txt".format(fi)
        #print(fname)
        t1.read(fname)
        loader = DataLoader(t1,batch_size=10,shuffle=False)
        #loader = DataLoader(t1,batch_size=100,shuffle=False)
        preds = process_crumble(t1,loader,model)
        #print(preds)
        if nbreaks:
            breaks_table.append((fname,t1.breaks_pos,t1.breaks_neg,preds))
        else:
            breaks_table.append((fname,t1.breaks_pos,[],preds))


    bdf = pd.DataFrame(breaks_table,columns=["text_id","breaks_pos","breaks_neg","predictions"])
    #bdf.to_csv("predictions_200breaks_nofinetuning_lr254_{0:.0f}.csv".format(time.time()))
    bdf.to_csv("predictions_allbreaks_nofinetuning_lr254_{0:.0f}.csv".format(time.time()))
