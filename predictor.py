import numpy as np
import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer, AutoTokenizer, DistilBertModel, AutoModelForMaskedLM, DistilBertTokenizer
import transformers

class BERT(nn.Module):

    def __init__(self, bert, max_length, num_classes, hidden_layers):
      
      super(BERT, self).__init__()

      self.bert = bert
      self.fcA1 = nn.Linear(hidden_layers,52)
      self.reluA =  nn.ReLU()
      self.dropoutA = nn.Dropout(0.1)
      self.fcA2 = nn.Linear(52,num_classes)
      self.softmaxA = nn.LogSoftmax(dim=-1)

    #define the forward pass
    def forward(self, sent_id_1, mask_1): 
      sentence_vec = self.bert(sent_id_1, attention_mask=mask_1)[0]

      sentence_vec = sentence_vec[:,0,:]
      x = self.fcA1(sentence_vec)
      x = self.reluA(x)
      x = self.dropoutA(x)
      x = self.fcA2(x)

      return x

def test(model, test_dataloader):
  
    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    total_preds = None

    # iterate over batches
    for step,batch in enumerate(test_dataloader):

        # push the batch to gpu
        batch = [t for t in batch]

        sent_id, mask = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            predsA = model(sent_id, mask)
            #preds = preds.squeeze()
            #print("preds shape: ", preds.shape)

            predsA = predsA.detach().cpu().numpy()
            if total_preds is not None:
                total_preds = np.concatenate((total_preds,predsA))
            else:
                total_preds = predsA

    return total_preds

def predict(test_file, hidden_layers, lang, SCRATCH_FNAME):
    f = open(test_file)
    test_sentences = f.readlines()
    f.close()

    if lang == 'en':
        model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        bert = model_class.from_pretrained(pretrained_weights)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
        bert = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-uncased")

    test_sentences_tokenized = np.array([tokenizer.encode(x, add_special_tokens=True, max_length=40, truncation=True) for x in test_sentences])

    max_len = 40

    test_sentences_tokenized_padded = np.array([i + [0]*(max_len-len(i)) for i in test_sentences_tokenized])
    attention_mask_test = np.where(test_sentences_tokenized_padded != 0, 1, 0)

    input_ids_test = torch.tensor(test_sentences_tokenized_padded) 
    attention_mask_test = torch.tensor(attention_mask_test)

    # wrap tensors
    test_data = TensorDataset(input_ids_test, attention_mask_test)
    # sampler for sampling the data during training
    test_sampler = RandomSampler(test_data)
    # dataLoader for train set
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

    ############### LOAD BERT ##########################
    model = BERT(bert, max_len, 2, hidden_layers)

    ############### predict ##########################
    #load weights of best model
    path = SCRATCH_FNAME
    model = model.cpu()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # get predictions for test data
    with torch.no_grad():
        total_preds = test(model, test_dataloader)
        #print(total_preds)
        total_preds = np.argmax(total_preds, axis = 1)
        print(total_preds)
    return total_preds

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    test_file = sys.argv[1] #'sample.txt'
    hidden_layers = int(sys.argv[2]) #768
    lang = sys.argv[3] #'en'
    SCRATCH_FNAME = sys.argv[4] #'urgency_en.pt'

    total_preds = predict(test_file, hidden_layers, lang, SCRATCH_FNAME)

