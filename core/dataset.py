import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from .utils import tien_xu_li
class LexDataset(Dataset):
    def __init__(   self,
                    data_path,
                    tokenizer,
                    modeltype = "t5",
                    batch = 256,
                    src_max_token_len = 256,
                    trg_max_token_len = 256,augumentator = None):
        super().__init__()

        self.tokenizer = tokenizer
        self.modeltype = modeltype
        self.src_max_token_len = src_max_token_len
        self.trg_max_token_len = trg_max_token_len
        self.augumentator =augumentator
        self.data = list()

        dataframe = pd.read_csv(data_path)

        try:
            dataframe = dataframe[["original", "normalized"]]
        except:
            dataframe = dataframe[["ceg", "norm"]]
        
        dataframe.columns = ["src", "trg"]

        self.prepare_io(dataframe, batch)

    def __len__(self):
        return len(self.data)
    
        
    def prepare_io(self, dataframe, batch):
        self.src = list(dataframe['src'])
        self.trg = list(dataframe['trg'])
        src_ids, trg_ids, src_masks, trg_masks = self.encoding(dataframe, batch)

        with tqdm(desc='Indexing... ' , unit='it', total=len(dataframe)) as pbar:
            for index in range(len(dataframe)):
                src_id = torch.tensor(src_ids[index], dtype=torch.int32)
                trg_id = torch.tensor(trg_ids[index], dtype=torch.int32)
            
                src_attention_mask = torch.tensor(src_masks[index], dtype=torch.int32)
                label_attention_mask = torch.tensor(trg_masks[index], dtype=torch.int32)


                self.data.append({'input_ids': src_id.flatten(), 'labels': trg_id.flatten(),
                                "src_attention_mask":src_attention_mask.flatten(), "label_attention_mask": label_attention_mask.flatten(), })
                
                pbar.update()

    
    def encoding(self, dataframe, batch):
        src_ids = []
        trg_ids = []
        src_masks = []
        trg_masks = []
        with tqdm(desc='Encoding... ' , unit='it', total=int(np.ceil(len(dataframe)/batch))) as pbar:
            for i in range(0, len(dataframe), batch):
                srcs = [tien_xu_li(question.strip()) for question in list(dataframe['src'][i:i+batch])]
                if self.augumentator:
                    srcs = [self.augumentator(question) for question in srcs ]
                    
                if self.modeltype == "t5":
                    trgs = [self.tokenizer.pad_token + ans.strip() for ans in list(dataframe['trg'][i:i+batch])]
                else:
                    trgs = [tien_xu_li(ans.strip()) for ans in list(dataframe['trg'][i:i+batch])]

                src_encoding = self.tokenizer(srcs,
                                                padding='max_length',
                                                max_length = self.src_max_token_len,
                                                truncation = True)
            
                trg_encoding = self.tokenizer(trgs,
                                                padding='max_length',
                                                max_length = self.trg_max_token_len,
                                                truncation = True)

                
                src_ids += src_encoding["input_ids"]
                src_masks += src_encoding["attention_mask"]

                trg_ids += trg_encoding["input_ids"]
                trg_masks += trg_encoding["attention_mask"]

                pbar.update()

        return src_ids, trg_ids, src_masks, trg_masks

    def __getitem__(self, index: int):
        return self.data[index]
