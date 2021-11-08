# -*- coding: utf-8 -*-
############## Finetuning BERT for Narrative Level Prediction ################

####Imports 
import glob
import datasets
from datasets import Dataset
import random
import torch
import transformers
from transformers import DataCollatorWithPadding #für das dynamische Padding
from torch.utils.data import DataLoader
from transformers import AdamW #Optimizer
from transformers import get_scheduler #Learning Rate Scheduler
from tqdm.auto import tqdm #für den Fortschrittsbalken
from datasets import list_metrics #Metrics fürs Testen des Modells
from datasets import load_metric

########################### FUNKTIONEN ###########################

# Ein paar dieser Funktionen (von denen zum BERT-Modell) orientieren sich an an diesem Tutorium: https://huggingface.co/course/chapter0?fw=pt
# Ich habe außerdem Pythonversion 3.7.1.1 verwendet
# Ganz unten werden dann die Funktionen aufgerufen (unter der Überschrift "das Programm ausführen")

######## Daten laden ########


def load_test_data():
  """"
  Lädt die Testdaten und packt alle Sätze in eine Liste: sents_test
  """
  docs_test = []
  sentences_test = []
  for file in files_test:
    fi = open(file)
    #print(fi)
    doc_sentences = fi.readlines()
    docs_test.append(doc_sentences)

  for sentence_list in docs_test:
    sentence_list.append("<BREAK/>\n") 
    for sentence in sentence_list:
      sentences_test.append(sentence)

  sents_test = [s for s in sentences_test if not ((len(s.strip()) == 0) or ("THE END" in s))]
  return sents_test


# test_dict = alle Trainingsdaten (nicht balanciert und nicht tokenisiert)
test_dict = {
    "sentence1": [],
    "sentence2": [],
    "labels": [],
    "idx": []}


### 1) Auf Absatzebene/ (ohne Kontext-Window) ###


## Testdaten
def par_level_label0_test(): 
  """
  Packt die Absatz-Paare in das test-dict. 
  Hier Absätze, die zusammengehören, also Label 0.
  """
  for i in range(0,len(sents_test),2):
    sent = sents_test[i]
    if not "BREAK" in sent and len(sent.split(' ')) < 500: 
        if i+1 in range(len(sents_test)):
            second_sent = sents_test[i+1]
            if not "BREAK" in second_sent and len(second_sent.split(' ')) < 500:
                test_dict["sentence1"].append(sent)
                test_dict["sentence2"].append(second_sent)
                test_dict["labels"].append(0) 
                test_dict["idx"].append(i) 

def par_level_label1_test():
  """
  Packt die Absatz-Paare in das test-dict. 
  Hier Absätze, die nicht zusammengehören, also Label 0.
  """
  for i in range(len(sents_test)): 
    sent = sents_test[i]
    if sent == "\n":
        continue
    elif "BREAK" in sent: 
        test_dict["sentence1"].append(sents_test[i-1])
        test_dict["sentence2"].append(sents_test[i+1])
        test_dict["labels"].append(1)
        test_dict["idx"].append(i) 


#######
### 2) Auf Wortebene/ mit Kontext-Window ###

## Testdaten
def word_level_label0_test():
    """
    Packt, gegeben ein Kontext-Windwow, entsprechend viele Wörter in das 
    test_dict. Hier Sätze, die zusammengehören, also Label 0.
    """
    for i in range(len(sents_test)):
        if "<BREAK/>" in sents_test[i]:
            continue
        if not "<BREAK/>" in sents_test[i+1]:
            text_words1 = sents_test[i].split()
            text_words2 = sents_test[i+1].split()
            text_words1_len = len(text_words1)

            if WINDOW_SIZE > text_words1_len:
                start_index_1 = 0
            else:
                start_index_1 = text_words1_len - WINDOW_SIZE

            words1 = text_words1[start_index_1:text_words1_len]
            words2 = text_words2[0:WINDOW_SIZE]

            test_dict["sentence1"].append(" ".join(words1))
            test_dict["sentence2"].append(" ".join(words2))
            test_dict["labels"].append(0)
            test_dict["idx"].append(i)

def word_level_label1_test():
    """
    Packt, gegeben ein Kontext-Windwow, entsprechend viele Wörter in das 
    test_dict. Hier Sätze, die nicht zusammengehören, also Label 1.
    """
    for i in range(len(sents_test)):
        if i == len(sents_test) - 1:
            break

        if "<BREAK/>" in sents_test[i]:
            text_words1 = sents_test[i - 1].split()
            text_words2 = sents_test[i + 1].split()
            text_words1_len = len(text_words1)

            if WINDOW_SIZE > text_words1_len:
                start_index_1 = 0
            else:
                start_index_1 = text_words1_len - WINDOW_SIZE

            words1 = text_words1[start_index_1:text_words1_len]
            words2 = text_words2[0:WINDOW_SIZE]

            test_dict["sentence1"].append(" ".join(words1))
            test_dict["sentence2"].append(" ".join(words2))
            test_dict["labels"].append(1)
            test_dict["idx"].append(i)


def sample_paragraph_level():
  """
  Ruft alle Funktionen auf, die für das Datasampling auf Absatz-Ebene 
  gebraucht werden
  """
  par_level_label0_test()
  par_level_label1_test()


## Testdaten
#Dictionary, dass alle Test-Items mit Label 1 enthält
test_dict_ones = {
    "sentence1": [],
    "sentence2": [],
    "labels": [],
    "idx":[]}

#Dictionary, dass alle Test-Items mit Label 0 enthält
test_dict_zeros = { 
    "sentence1": [],
    "sentence2": [],
    "labels": [],
    "idx":[]}

#Dictionary, das zufällig ausgewählte Daten mit Label 0 enthält
#und davon genau so viele  wie test_dict_ones enthält
test_dict_new = {
        "sentence1": [],
        "sentence2": [],
        "labels": [],
        "idx": []}



def keep_unbalanced_test_data():
  """
  Lässt die Testdaten unbalanciert. Ändert nur den Namen in test_dict_new, weil
  der Tokenizer später ein dict mit diesem Namen erwartet. 
  """
  for i in range(0, len(test_dict["labels"])):
    test_dict_new["sentence1"].append(test_dict["sentence1"][i])
    test_dict_new["sentence2"].append(test_dict["sentence2"][i])
    test_dict_new["labels"].append(test_dict["labels"][i])
    test_dict_new["idx"].append(test_dict["idx"][i])
  


######## Daten für das Modell finalisieren ########

def tokenize_function(example):
  """
  Tokenisiert die Satzpaare. 
  Gibt ein dictionary mit den keys input_ids, attention_mask und token_type_ids zurück.
  Fügt den Sätzen außerdem das CLS- und die SEP-Token hinzu.
  """
  return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def create_dataset():
  """
  Nimmt die Trainings- und Testdaten, erstellt einen Datensatz,
  shuffelt diesen Datensatz und tokenisiert diesen Datensatz mit der tokenize_Funktion().
  """
  # aus den Dictionaries jeweils ein huggingface Datasetobjekt machen:
  test_data = Dataset.from_dict(test_dict_new)
  # die Daten shufflen:
  test_data = test_data.shuffle()
  print(len(test_data),"len test data")
  # zu einem Datensatz zusammenfügen:
  data = datasets.DatasetDict({"test":test_data}) 
  # diese Daten tokenisieren:
  tokenized_data = data.map(tokenize_function, batched=True) 
  # Werte entfernen, die das Modell nicht erwartet:
  tokenized_data = tokenized_data.remove_columns(["sentence1", "sentence2", "idx"]) 
  # Pytorch-Tensoren (und nicht Listen) zurückgeben:
  tokenized_data.set_format("torch")
  return tokenized_data

def create_dataloader():
  """
  Nimmt die tokenisierten Daten und erstellt train_loader und test_loader 
  für das Modell. Padded außerdem die Daten (dynamisch) mit dem data_collator.
  """
  tokenized_data= create_dataset()
  test_dataloader = DataLoader(tokenized_data["test"], shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator)
  return test_dataloader


######## Zum Modell ######## 


def load_metrics():
  """
  Lädt Metriken aus der huggingface Bibliothek. 
  """
  metric1 = load_metric("accuracy")
  metric2 = load_metric("precision")
  metric3 = load_metric("recall")
  metric4 = load_metric("f1")
  return metric1, metric2, metric3, metric4

def test_model():
  """
  Testet das Modell und berechnet die geladenen Metriken.
  """
  print("--------MODELL TESTEN--------")

  batch_count = 0
  metric1, metric2, metric3, metric4 = load_metrics()

  model.eval()
  for batch in test_dataloader:
      batch_count +=1
      print(batch_count)
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
        outputs = model(**batch)
    
      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)

      ### Zur Kontrolle ausprinten:
      #print("-------")
      #print("Batch:", batch_count)
      #print("Labels:", batch["labels"])
      #print("Predictions:",predictions)
      #print("Sentences:")
      #for i in batch["input_ids"]:
        #print(tokenizer.decode(i))
        
      ###########
      metric1.add_batch(predictions=predictions, references=batch["labels"])
      metric2.add_batch(predictions=predictions, references=batch["labels"])
      metric3.add_batch(predictions=predictions, references=batch["labels"])
      metric4.add_batch(predictions=predictions, references=batch["labels"])
  
  print("accuracy:", metric1.compute()["accuracy"])
  print("precision:", metric2.compute()["precision"])
  print("recall:", metric3.compute()["recall"])
  print("f1:", metric4.compute()["f1"])



################################################
############# das Programm ausführen ############

#### Daten laden 
path_test= "corpus1/test" 

files_test = glob.glob(path_test + '/*.txt')

#### Daten vorbereiten/ Sätze in Liste packen
print("Daten vorbereiten/ Sätze in Liste packen...")
sents_test = load_test_data()
print(len(sents_test),"test sentences")

#### Datasampling: 1) oder 2) (das jeweils andere dann auskommentieren)
print("Datasampling...")
## 1) Daten auf Absatz-Ebene samplen (d.h. ganze Absätze, ohne WINDOW_SIZE, maximal 500 (wegen BERT))
sents_test= sents_test[:-1]
sample_paragraph_level()

## 2) Daten auf Wort-Ebene samplen (gegeben eine WINDOW_SIZE):
#WINDOW_SIZE = 250
#sample_word_level()

keep_unbalanced_test_data()

#### Tokenisierer und data_collator laden
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # um die Daten dynamisch zu padden

#### Daten tokenisieren, shuffeln, Datenloader für das Modell fertigmachen
print("Daten tokenisieren, Datenloader fertig machen...")
BATCH_SIZE = 10
test_dataloader = create_dataloader()

#### Das Modell instantiieren:
print("Modell instantiieren...")
#model = transformers.BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

model = transformers.AutoModel.from_pretrained("finetuned_models/ae_v1")


#### GPU verwenden
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
print("Device:", device)


#### das Modell testen/evaluieren:
test_model()

#################################
print("-------------------------")
print("BATCH_SIZE:", BATCH_SIZE)
#print("N_EPOCHS:", N_EPOCHS)
print("WINDOW_SIZE:", WINDOW_SIZE)

