from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers
import torch
import sys
sys.path.insert(0,'..')
from config import Config

config = Config()
def get_dataloader(a_s):
    input_ids = torch.tensor(a_s['input_ids'])
    attention_masks = torch.tensor(a_s['attention_masks'])
    labels = torch.tensor(a_s['labels'][0])
    print(input_ids.shape)
    print(attention_masks.shape)
    print(labels.shape)
    train_data = TensorDataset(input_ids, attention_masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    return train_dataloader


def tokenize_data(ds):
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  tokenizer = transformers.RobertaTokenizer.from_pretrained(config.model)
  a_s = {'input_ids': [], 'attention_masks': [], 'labels': []}
  for sent in ds['text']:
    a  = tokenizer.encode(sent, 
                    max_length=config.seq_length, 
                    padding=True,
                    truncation=True,
                    add_special_tokens = True)

    a_s['input_ids'].append(a)
    a_s['labels'].append(ds['label'])
  a_s['input_ids'] = pad_sequences(a_s['input_ids'], maxlen=config.seq_length, dtype="long", 
                          value=0, truncating="post", padding="post")
  a_s = get_masks(a_s)
  return a_s

def get_masks(ds_dict):
  # Create attention masks
  train_masks = []
  # For each sentence...
  for sent in ds_dict['input_ids']:
      # Create the attention mask.
      #   - If a token ID is 0, then it's padding, set the mask to 0.
      #   - If a token ID is > 0, then it's a real token, set the mask to 1.
      att_mask = [int(token_id > 0) for token_id in sent]
      
      # Store the attention mask for this sentence.
      ds_dict['attention_masks'].append(att_mask)
  return ds_dict


