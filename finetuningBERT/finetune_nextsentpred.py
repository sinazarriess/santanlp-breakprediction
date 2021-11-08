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

def load_train_data():
  """ 
  Lädt die Trainingsdaten und packt alle Sätze in eine Liste: sents_train
  """
  docs_train = []
  sentences_train = []
  for file in files_train:
    fi = open(file)
    #print(fi)
    doc_sentences = fi.readlines()
    docs_train.append(doc_sentences)

  for sentence_list in docs_train:
    sentence_list.append("<BREAK/>\n") 
    for sentence in sentence_list:
      sentences_train.append(sentence)

  sents_train = [s for s in sentences_train if not ((len(s.strip()) == 0) or ("THE END" in s))]
  return sents_train

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


######## Data sampling ########

# train_dict = alle Trainingsdaten (nicht balanciert und nicht tokenisiert)
train_dict = {
    "sentence1": [],
    "sentence2": [],
    "labels": [],
    "idx": []}

# test_dict = alle Trainingsdaten (nicht balanciert und nicht tokenisiert)
test_dict = {
    "sentence1": [],
    "sentence2": [],
    "labels": [],
    "idx": []}


### 1) Auf Absatzebene/ (ohne Kontext-Window) ###

## Trainingsdaten 
def par_level_label0_train():
  """
  Packt die Absatz-Paare in das train-dict. 
  Hier Absätze, die zusammengehören, also Label 0.
  """
  for i in range(0,len(sents_train),2): 
    sent = sents_train[i]
    if not "BREAK" in sent and len(sent.split(' ')) < 500: 
        if i+1 in range(len(sents_train)): 
            second_sent = sents_train[i+1]
            if not "BREAK" in second_sent and len(second_sent.split(' ')) < 500:
                train_dict["sentence1"].append(sent)
                train_dict["sentence2"].append(second_sent)
                train_dict["labels"].append(0) 
                train_dict["idx"].append(i) 

def par_level_label1_train():
  """
  Packt die Absatz-Paare in das train-dict. 
  Hier Absätze, die nicht zusammengehören, also Label 0.
  """
  for i in range(len(sents_train)):
    sent = sents_train[i]
    if sent == "\n": 
        continue
    elif "BREAK" in sent: 
        #print(sent)
        train_dict["sentence1"].append(sents_train[i-1])
        train_dict["sentence2"].append(sents_train[i+1])
        train_dict["labels"].append(1) 
        train_dict["idx"].append(i) 

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

def sample_paragraph_level():
  """
  Ruft alle Funktionen auf, die für das Datasampling auf Absatz-Ebene 
  gebraucht werden
  """
  par_level_label0_train()
  par_level_label1_train()
  par_level_label0_test()
  par_level_label1_test()



#######
### 2) Auf Wortebene/ mit Kontext-Window ###

## Trainingsdaten
def word_level_label0_train():
  """
  Packt, gegeben ein Kontext-Windwow, entsprechend viele Wörter in das 
  train_dict. Hier Sätze, die zusammengehören, also Label 0.
  """
  for i in range(len(sents_train)):
    if "<BREAK/>" in sents_train[i]:
      continue
    if not "<BREAK/>" in sents_train[i+1]:
      text_words1 = sents_train[i].split()
      text_words2 = sents_train[i+1].split()
      text_words1_len = len(text_words1)

      if WINDOW_SIZE > text_words1_len:
        start_index_1 = 0
      else:
        start_index_1 = text_words1_len - WINDOW_SIZE

      words1 = text_words1[start_index_1:text_words1_len]
      words2 = text_words2[0:WINDOW_SIZE]

      train_dict["sentence1"].append(" ".join(words1))
      train_dict["sentence2"].append(" ".join(words2))
      train_dict["labels"].append(0)
      train_dict["idx"].append(i)

def word_level_label1_train():
  """
  Packt, gegeben ein Kontext-Windwow, entsprechend viele Wörter in das 
  train_dict. Hier Sätze, die nicht zusammengehören, also Label 1.
  """
  for i in range(len(sents_train)):
      if i == len(sents_train) - 1:
        break

      if "<BREAK/>" in sents_train[i]:
        text_words1 = sents_train[i - 1].split()
        text_words2 = sents_train[i + 1].split()
        text_words1_len = len(text_words1)

        if WINDOW_SIZE > text_words1_len:
          start_index_1 = 0
        else:
          start_index_1 = text_words1_len - WINDOW_SIZE

        words1 = text_words1[start_index_1:text_words1_len]
        words2 = text_words2[0:WINDOW_SIZE]

        train_dict["sentence1"].append(" ".join(words1))
        train_dict["sentence2"].append(" ".join(words2))
        train_dict["labels"].append(1)
        train_dict["idx"].append(i)

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


def sample_word_level():
  """
  Ruft alle Funktionen auf, die für das Datasampling auf Wortebene 
  (also mit Kontext-Window) gebraucht werden.
  """
  word_level_label0_train()
  word_level_label1_train()
  word_level_label0_test()
  word_level_label1_test()



######## Daten ausbalancieren ########

### Undersampling ###

## Trainingsdaten
#Dictionary, dass alle Training-Items mit Label 1 enthält
train_dict_ones = {
    "sentence1": [],
    "sentence2": [],
    "labels": [],
    "idx":[]}

#Dictionary, dass alle Training-Items mit Label 0 enthält
train_dict_zeros = { 
    "sentence1": [],
    "sentence2": [],
    "labels": [],
    "idx":[]}

#Dictionary, das zufällig ausgewählte Daten mit Label 0 enthält
#und davon genau so viele die Anzahl in  train_dict_ones 
train_dict_new = {
        "sentence1": [],
        "sentence2": [],
        "labels": [],
        "idx": []}

def undersample_data_train():
  """
  Balanciert die Trainingsdaten so, dass gleich viele Daten mit Label 0 wie 
  Daten mit Label 1 enthalten sind. Dies anhand von "Undersampling", das heißt,
  von den Labels, die in der Überzahl da sind, werden so viele zufällig 
  aussoriert, bis die Labels in der gleichen Zahl vertreten sind.
  """
  for i in range(0, len(train_dict["labels"])):
    if train_dict["labels"][i] == 1:
      train_dict_ones["sentence1"].append(train_dict["sentence1"][i])
      train_dict_ones["sentence2"].append(train_dict["sentence2"][i])
      train_dict_ones["labels"].append(train_dict["labels"][i])
      train_dict_ones["idx"].append(train_dict["idx"][i])
    else:
      train_dict_zeros["sentence1"].append(train_dict["sentence1"][i])
      train_dict_zeros["sentence2"].append(train_dict["sentence2"][i])
      train_dict_zeros["labels"].append(train_dict["labels"][i])
      train_dict_zeros["idx"].append(train_dict["idx"][i])    
  #print("Einsen vorher (train_dict_ones):", len(train_dict_ones["labels"])) 
  #print("Nullen vorher(train_dict_zeros):", len(train_dict_zeros["labels"])) 

  for x in range(0, len(train_dict_zeros["sentence1"])):
    if x == len(train_dict_ones["labels"]): #damit so viele Nullen, wie Einsen
      break
  # so zufällig Daten aus train_dict_zeros aussortieren:
    random_index = random.randint(0, len(train_dict_zeros["sentence1"])-1)
    train_dict_new["sentence1"].append(train_dict_zeros["sentence1"].pop(random_index))
    train_dict_new["sentence2"].append(train_dict_zeros["sentence2"].pop(random_index))
    train_dict_new["labels"].append(train_dict_zeros["labels"].pop(random_index))
    train_dict_new["idx"].append(train_dict_zeros["idx"].pop(random_index))
  #print("Nullen, neu (nach Aussortierung):", len(train_dict_new["labels"]))

  ## die beiden (nun gleich langen) Dictionaries wieder zusammenfügen bzw. das eine in das andere packen:
  for i in range(0, len(train_dict_ones["sentence1"])):
    train_dict_new["sentence1"].append(train_dict_ones["sentence1"][i])
    train_dict_new["sentence2"].append(train_dict_ones["sentence2"][i])
    train_dict_new["labels"].append(train_dict_ones["labels"][i])
    train_dict_new["idx"].append(train_dict_ones["idx"][i])
  #print("Nullen + Einsen gemeinsam neu:", len(train_dict_new["labels"])) 


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

def undersample_data_test():
  """
  Balanciert die Testdaten so, dass gleich viele Daten mit Label 0 wie 
  Daten mit Label 1 enthalten sind. Dies anhand von "Undersampling", das heißt,
  von den Labels, die in der Überzahl da sind, werden so viele zufällig 
  aussoriert, bis die Labels in der gleichen Zahl vertreten sind.
  """
  for i in range(0, len(test_dict["labels"])):
    if test_dict["labels"][i] == 1:
      test_dict_ones["sentence1"].append(test_dict["sentence1"][i])
      test_dict_ones["sentence2"].append(test_dict["sentence2"][i])
      test_dict_ones["labels"].append(test_dict["labels"][i])
      test_dict_ones["idx"].append(test_dict["idx"][i])
    else:
      test_dict_zeros["sentence1"].append(test_dict["sentence1"][i])
      test_dict_zeros["sentence2"].append(test_dict["sentence2"][i])
      test_dict_zeros["labels"].append(test_dict["labels"][i])
      test_dict_zeros["idx"].append(test_dict["idx"][i])  
  #print("Einsen vorher (test_dict_ones):", len(test_dict_ones["labels"])) #passt
  #print("Nullen vorher (test_dict_zeros):", len(test_dict_zeros["labels"])) #passt

  for x in range(0, len(test_dict_zeros["sentence1"])):
    if x == len(test_dict_ones["labels"]): #damit so viele Nullen, wie Einsen
      break
  # so zufällig Daten aus test_dict_zeros aussortieren:
    random_index = random.randint(0, len(test_dict_zeros["sentence1"])-1)
    test_dict_new["sentence1"].append(test_dict_zeros["sentence1"].pop(random_index))
    test_dict_new["sentence2"].append(test_dict_zeros["sentence2"].pop(random_index))
    test_dict_new["labels"].append(test_dict_zeros["labels"].pop(random_index))
    test_dict_new["idx"].append(test_dict_zeros["idx"].pop(random_index))
  #print("Nullen, neu (nach Aussortierung):", len(test_dict_new["labels"]))

  ## die beiden (nun gleich langen) Dictionaries wieder zusammenfügen bzw. das eine in das andere packen:
  for i in range(0, len(test_dict_ones["sentence1"])):
    test_dict_new["sentence1"].append(test_dict_ones["sentence1"][i])
    test_dict_new["sentence2"].append(test_dict_ones["sentence2"][i])
    test_dict_new["labels"].append(test_dict_ones["labels"][i])
    test_dict_new["idx"].append(test_dict_ones["idx"][i])
  #print("Nullen + Einsen gemeinsam neu:", len(test_dict_new["labels"]))


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
  

def undersample_data_train_and_test():
  """
  Ruft die Funktionen auf, um sowohl Trainings- als auch Testdaten zu balancieren .
  """
  undersample_data_train()
  undersample_data_test()

def undersample_data_only_train():
  """
  Ruft die Funktionen auf, um nur die Trainingsdaten zu balancieren.
  Die Testdaten bleiben hier unbalanciert.
  """
  undersample_data_train()
  keep_unbalanced_test_data()


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
  train_data = Dataset.from_dict(train_dict_new)
  test_data = Dataset.from_dict(test_dict_new)
  # die Daten shufflen:
  train_data = train_data.shuffle()
  test_data = test_data.shuffle()
  # zu einem Datensatz zusammenfügen:
  data = datasets.DatasetDict({"train":train_data, "test":test_data}) 
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
  train_dataloader = DataLoader(tokenized_data["train"], shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator)
  test_dataloader = DataLoader(tokenized_data["test"], shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator)
  return train_dataloader, test_dataloader


######## Zum Modell ######## 

def define_training_steps():
  """
  Berechnet die Anzahl der Trainingsschritte: 
  die Anzahl der Epochen multipliziert mit der Anzahl 
  der Trainingsbatches (d.h. mit der Länge von train_dataloader)
  """
  n_training_steps = N_EPOCHS * len(train_dataloader) 
  return n_training_steps

def learning_rate_scheduler():
  """
  Nutzt get_scheduler() von hugging face um eine Lernrate zu definieren.
  Die standardmäßig verwendete Lernrate ist ein linearer Abstieg 
  vom Maximalwert (5e-5) auf 0.  
  """
  lr_scheduler = get_scheduler(
      "linear", #linearer Abstieg
      optimizer=optimizer, 
      num_warmup_steps=0,  
      num_training_steps=n_training_steps) 
  return lr_scheduler

def create_progress_bar():
  """
  Erstellt einen Fortschrittbalken mit tqdm.
  """
  progress_bar = tqdm(range(n_training_steps))
  return progress_bar


def train_model():
  """
  Trainiert das Modell und berechnet den durchschnittlichen Trainingsloss 
  pro Epoche.
  """
  print("--------MODELL TRAINIEREN--------")
  progress_bar = create_progress_bar()
  #loss_values = []
  epoch_count = 0

  model.train()
  for epoch in range(N_EPOCHS):
    total_loss = 0 
    epoch_count +=1

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
      
        outputs = model(**batch)
        loss = outputs.loss #Loss bestimmen
        total_loss += loss.item() 
        #print("LOSS:", loss.item())
        #print("TOTAL LOSS:", total_loss) 

        loss.backward() # Back Propagation
        optimizer.step() # Gewichte updaten 
        lr_scheduler.step()
        optimizer.zero_grad() #die akkumulierten Gewichte löschen

        progress_bar.update(1)

    train_loss = total_loss/len(train_dataloader) 
    #loss_values.append(train_loss) 
    print("Epoche", epoch_count)
    print("Durchschnittlicher Training Loss: ",train_loss)  

    model.save_pretrained("./finetuned_models/ae_v1")

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
path_train= "../../santanlp-corpus/corpus1/train" 
path_test= "../../santanlp-corpus/corpus1/test" 

files_train = glob.glob(path_train + '/*.txt')
files_test = glob.glob(path_test + '/*.txt')

#### Daten vorbereiten/ Sätze in Liste packen
print("Daten vorbereiten/ Sätze in Liste packen...")
sents_train = load_train_data()
sents_test = load_test_data()


#### Datasampling: 1) oder 2) (das jeweils andere dann auskommentieren)
print("Datasampling...")
## 1) Daten auf Absatz-Ebene samplen (d.h. ganze Absätze, ohne WINDOW_SIZE, maximal 500 (wegen BERT))
sents_train= sents_train[:-1] 
sents_test= sents_test[:-1]
sample_paragraph_level()

## 2) Daten auf Wort-Ebene samplen (gegeben eine WINDOW_SIZE):
#WINDOW_SIZE = 250
#sample_word_level()


#### Daten ausbalancieren:  1) oder 2) (das jeweils andere dann auskommentieren)
print("Daten ausbalancieren...")
## 1) Trainings- und Testdaten balancieren
undersample_data_train_and_test()

## 2) nur die Trainingsdaten ausbalancieren 
#undersample_data_only_train() 
 

#### Tokenisierer und data_collator laden
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # um die Daten dynamisch zu padden

#### Daten tokenisieren, shuffeln, Datenloader für das Modell fertigmachen
print("Daten tokenisieren, Datenloader fertig machen...")
BATCH_SIZE = 10
train_dataloader, test_dataloader = create_dataloader()

#### Das Modell instantiieren:
print("Modell instantiieren...")
model = transformers.BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

#### Den Optimizer festlegen:
optimizer = AdamW(model.parameters(), lr=5e-5)

#### Den Learning Rate Scheduler einrichten/ die Lernrate definieren:
N_EPOCHS = 3
n_training_steps = define_training_steps()
lr_scheduler= learning_rate_scheduler()

#### GPU verwenden
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
print("Device:", device)

#### das Modell trainieren:
train_model()

#### das Modell testen/evaluieren:
test_model()

#################################
print("-------------------------")
print("BATCH_SIZE:", BATCH_SIZE)
#print("N_EPOCHS:", N_EPOCHS)
print("WINDOW_SIZE:", WINDOW_SIZE)

