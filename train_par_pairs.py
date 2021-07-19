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


def finetune_on_crumble(traintext,train_loader,model):

    print("Initializing GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    # Parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
    ]

    # Optimizer
    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    print("Making model GPU compatible")
    model = model.to(device)

    epochs = 4

    print('Starting training...')

    loss_values = []
    for epoch_i in range(0, epochs):
        print('Epoch ', epoch_i)

        model.train()

        t0 = time.time()

        total_loss = 0

        for step, batch in enumerate(train_loader):

            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


            batch = tuple(t.to(device) for t in batch)
            b_text_ids, b_input_ids, b_seg_ids, b_attention_masks, b_labels = batch

            optimizer.zero_grad()

            outputs = model(b_input_ids, token_type_ids=b_seg_ids, attention_mask=b_attention_masks, next_sentence_label=b_labels)

            loss = outputs[0]

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))


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
        fname = "../santanlp-corpus/corpus1/train/{0:0=3d}.txt".format(fi)
        #print(fname)
        t1.read(fname)
        loader = DataLoader(t1,n=50,batch_size=100,shuffle=False)
        finetune_on_crumble(t1,loader,model)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print('Saved model to ' + output_dir)
