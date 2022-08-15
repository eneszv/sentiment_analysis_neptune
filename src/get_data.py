from datasets import load_dataset
import random

def get_data(n_samples = 10):

    dataset = load_dataset("imdb")['train']

    idx = random.sample(range(len(dataset['train']['text'])), n_samples)

    data_dict = {
        'text': [dataset['train']['text'][i] for i in idx],
        'labels': [dataset['train']['label'][i] for i in idx]
    }
    
    return data_sample


