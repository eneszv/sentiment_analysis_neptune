from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pandas as pd
from time import gmtime, strftime
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def save_res(res_dict, out_dir, mode):
    
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    df = pd.DataFrame(os.path.join(out_dir, f'{mode}_pred_{time}.csv'))


def run_model(data_sample, 
              model_name="distilbert-base-uncased-finetuned-sst-2-english",
              out_dir='s3://experimental/sentiment-analyis-data/live'):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    res_dict = {'pred':[], 'act':[]}
    for i in range(len(data_sample['text'])):
    
        inp = tokenizer(data_sample['text'][i], return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            logits = model(**inp).logits
            
        pred = logits.argmax().item()
        res_dict['pred'].append(pred)
        res_dict['act'].append(data_sample['labels'][i])
        
    save_res(res_dict, out_dir, 'live')
    
    return res

def run_shadow_model(data_sample, 
                     model_name="cardiffnlp/twitter-roberta-base-sentiment", 
                     out_dir='s3://experimental/sentiment-analyis-data/shadow'):
                     
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    res_dict = {'pred':[], 'act':[]}
    for i in range(len(data_sample['text'])):
    
        inp = tokenizer(data_sample['text'][i], return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            logits = model(**inp).logits
        
        if logits[0] > logits[2]:
            pred = 0
        else:
            pred = 1

        res_dict['pred'].append(pred)
        res_dict['act'].append(data_sample['labels'][i])
        
    save_res(res_dict, out_dir, 'shadow')
    
    return res                    