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
import argparse
from test_par_pairs import TextCrumble, process_crumble


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--in', dest='inputfile')
    parser.add_argument('--out', dest='outputfile')
    parser.add_argument('--model', dest='model',default="pretrained")
    #parser.add_argument('--context', dest='context', type=int, default=254)

    args = parser.parse_args()


    batch_i = 0
    breaks_table = []
    #nbreaks = 200
    nbreaks = None

    print("cuda",torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device",device)

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


    print("model: ",args.model)
    if args.model == "pretrained":
        model = transformers.BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    else:
        model = transformers.BertForNextSentencePrediction.from_pretrained(args.model)

    model = model.to(device)
    model.eval()

    t1 = TextCrumble(tokenizer,n=nbreaks,window=254)
    fname = args.inputfile
    t1.read(fname)
    loader = DataLoader(t1,batch_size=10,shuffle=False)
    preds = process_crumble(t1,loader,model,device=device)

    bdf = pd.DataFrame(preds,columns=["sent_id","gold_label","pred_label","prob","sent"])
    bdf.to_csv(args.outputfile+".noft_lr254.csv")

    t1 = TextCrumble(tokenizer,n=nbreaks,window=154)
    fname = args.inputfile
    t1.read(fname)
    loader = DataLoader(t1,batch_size=10,shuffle=False)
    preds = process_crumble(t1,loader,model,device=device)

    bdf = pd.DataFrame(preds,columns=["sent_id","gold_label","pred_label","prob","sent"])
    bdf.to_csv(args.outputfile+".noft_lr154.csv")


    t1 = TextCrumble(tokenizer,n=nbreaks,window=54)
    fname = args.inputfile
    t1.read(fname)
    loader = DataLoader(t1,batch_size=10,shuffle=False)
    preds = process_crumble(t1,loader,model,device=device)

    bdf = pd.DataFrame(preds,columns=["sent_id","gold_label","pred_label","prob","sent"])
    bdf.to_csv(args.outputfile+".noft_lr54.csv")
