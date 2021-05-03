import numpy as np
import pandas as pd
import torch, sys
import torch.nn as nn
from sklearn.model_selection import train_test_split
import transformers
from transformers import AutoTokenizer, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences
import re, emoji
device = torch.device("cuda")

cuda_status = 0

############### PREEPROCESS TWEET ##########################
def preprocess(sentence):
    sentence = re.sub(r'http\S+', ' ', sentence)
    words = sentence.split()
    words = [word for word in words if ('.com' not in word and 'www' not in word and '\\\\' not in word and '\\u' not in word and '//' not in word)]
    sentence = " ".join(words)
    sentence = re.sub("([!]){1,}", " ! ", sentence)
    sentence = re.sub("([.]){1,}", " . ", sentence)
    sentence = re.sub("([?]){1,}", " ? ", sentence)
    sentence = re.sub("([;]){1,}", " ; ", sentence)
    sentence = re.sub("([:]){1,}", " : ", sentence)
    sentence = re.sub("([,]){1,}", " : ", sentence)
    for c in sentence:
        if c in emoji.UNICODE_EMOJI:
            sentence = re.sub(c, " ", sentence)
            #sentence = re.sub(c, " "+emoji.demojize(c).replace("_"," ")+" ", sentence)
    words = sentence.split()
    words = [word for word in words if (word[0] != '#' and '@' not in word and '.com' not in word and 'www' not in word and '\\\\' not in word and '\\u' not in word and '//' not in word)]
    sentence = " ".join(words)
    sentence = re.sub('_', " ", sentence)
    sentence = re.sub('\s+', ' ', sentence)
    sentence = re.sub('&', " and ", sentence)
    #sentence = sentence.encode('ascii', 'ignore')
    return sentence

############### CREATE TOKENIZER ##########################

from transformers import AutoTokenizer, AutoModelForMaskedLM  
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

#define a batch size
batch_size = 32

############### BERT ##########################
class mBERT(nn.Module):

    def __init__(self, bert, num_labels, hidden_layers, drop=0.1):
      
      super(mBERT, self).__init__()

      self.bert = bert
      self.dropout = nn.Dropout(p=drop, inplace=False)
      self.linear1 = nn.Linear(hidden_layers, 16)
      self.linear2 = nn.Linear(16, num_labels)

    #define the forward pass
    def forward(self, sent_id_1, mask_1):
      
      out = self.bert(sent_id_1, token_type_ids=None, attention_mask=mask_1)

      sent_vec = out.hidden_states[-1][:,0,:]

      sent_vec = self.dropout(sent_vec)
      sent_vec = self.linear1(sent_vec)
      sent_vec = self.linear2(sent_vec)

      return sent_vec

def test(model, test_dataloader):
  
    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    all_labels = []
    confidence_scores = []

    # iterate over batches
    for step,batch in enumerate(test_dataloader):

        # push the batch to gpu
        if torch.cuda.is_available():
          batch = [t.to(device) for t in batch]
        else:
          batch = [t for t in batch]
        #batch = [t for t in batch]

        sent_id, mask = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)
            sigpreds = nn.Sigmoid()(preds)
            sigpreds=sigpreds.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            total_preds = np.argmax(preds, axis=1)
            all_labels.extend(total_preds)
            confidence_scores.extend(np.max(sigpreds, axis=1))

    return all_labels, confidence_scores

def create_loaders(seq, mask):
  # wrap tensors
  data = TensorDataset(seq, mask)
  # sampler for sampling the data during training
  sampler = RandomSampler(data)
  # dataLoader for train set
  dataloader = DataLoader(data, sampler=None, batch_size=batch_size)
  return dataloader

def tokenize_and_preserve_labels(sentence):
    tokenized_sentence = []

    for word in sentence:
        tokenized_word = tokenizer.tokenize(word)
        tokenized_sentence.extend(tokenized_word)

    tokenized_sentence = [x for x in tokenized_sentence if not x.isdigit()]
    
    return tokenized_sentence


def getData4Bert(data, tokenizer, MAX_LEN=100):

    sentences = data

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent)
        for sent in sentences
    ]

    tokenized_texts = [token_label_pair for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", value=0.0,
                              truncating="post", padding="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    return torch.tensor(input_ids), torch.tensor(attention_masks)

def predict(data, fname):

  data = [preprocess(x) for x in data]

  hidden_layers = 768
  test_seq, test_mask = getData4Bert(data, tokenizer)
  test_dataloader = create_loaders(test_seq, test_mask)

  ############### LOAD BERT ##########################
  bert = RobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels = 2, output_attentions = False, output_hidden_states = True, return_dict=True)

  ############### LOAD BERT ##########################
  model = mBERT(bert, 2, hidden_layers)
  # push the model to GPU

  if torch.cuda.is_available():
    model = model.to(device)
    model.load_state_dict(torch.load(fname))
  else:
    model = model.to('cpu')
    model.load_state_dict(torch.load(fname,map_location=torch.device('cpu')))

  # get predictions for test data
  with torch.no_grad():
      all_labels, confidence_scores = test(model, test_dataloader)

  return all_labels, confidence_scores

if __name__ == '__main__':
  # execute only if run as the entry point into the program
  test_file = sys.argv[1] #"offer_samples.txt"
  fname = sys.argv[2] #"offer.pt"
  fname_RESULT = sys.argv[3] #"result.txt"

  f = open(test_file)
  data = f.readlines()
  f.close()
  data = [x.strip() for x in data]

  labels, cscores = predict(data, fname)
  assert(len(labels) == len(data))
  f = open(fname_RESULT, 'w+')
  for index in range(len(labels)):
    f.write(str(labels[index])+","+str(round(cscores[index],4))+'\n')
  f.close()

