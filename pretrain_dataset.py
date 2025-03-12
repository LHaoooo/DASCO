import torch
import sys
sys.path.append('Text_encoder')
sys.path.append('PDQ')
from torch.utils.data import Dataset, DataLoader
import os
import torch
from transformers import BertTokenizer
import pdb
from tqdm import tqdm
import os
import pickle
import json

def get_span(target,input_ids,tokenizer):
    # 假设已有tokenizer和input_ids
    # 将待搜索字符串转换为token并获取其长度
    tgt_tokens = tokenizer.encode(target, add_special_tokens=False)
    tgt_token_len = len(tgt_tokens)
    input_ids_list=input_ids.tolist()
    start_pos = []
    end_pos = []
    for i in range(len(input_ids_list)-tgt_token_len+1):
        if input_ids_list[i:i+tgt_token_len] == tgt_tokens:
            is_subword = False
            #  检测下一个 token 是否是子词
            if i + tgt_token_len < len(input_ids_list):
                next_token = tokenizer.convert_ids_to_tokens(input_ids_list[i  + tgt_token_len])
                # 判断规则：Hugging Face 风格子词（##）、SentencePiece 风格（▁）
                is_subword = next_token.startswith(('##',  '▁'))
            if not is_subword:
                start_pos.append(i)
                end_pos.append(i + tgt_token_len - 1)
    return start_pos,end_pos

class pretrain_dataset(Dataset):
    def __init__(self,
                IE_tokenizer,
                PQ_former_tokenizer,
                data_path,                
                max_seq_len=512,
                num_query_token=32,
                SEP_token_id=2,
                split_token_id=187284,
                set_size=10,
                ):
        super().__init__()
        #init data
        self.data=[]
        filelist = os.listdir(data_path)
        data_filelist=[x for x in filelist if x.endswith("pkl")]
        self.data_path=[os.path.join(data_path,fl) for fl in data_filelist]
        label_filelist=[x for x in filelist if x.endswith("json")]
        label_filelist=[os.path.join(data_path,fl) for fl in label_filelist]

        self.set_size=set_size
        self.max_seq_len = max_seq_len
        self.num_query_token=num_query_token
        self.PQ_former_tokenizer=PQ_former_tokenizer
        self.IE_tokenizer=IE_tokenizer
        self.SEP_token_id=SEP_token_id
        self.split_token_id=split_token_id
        self.current_data_index=0

    def update_data(self):
        set_size=self.set_size
        start_idx=self.current_data_index
        end_idx=start_idx+set_size if start_idx+set_size<len(self.data_path)+1 else len(self.data_path)
        current_data=self.data_path[start_idx:end_idx]
        self.data=[]
        for path in tqdm(current_data,desc="data loading"):
            with open(path, 'rb') as f:
                temp=pickle.load(f)
                self.data.extend(temp)
        self.current_data_index=end_idx
        print("index here:",self.current_data_index)
    
    def is_end(self):
        return self.current_data_index==len(self.data_path)

    def reset(self):
        self.current_data_index=0

    def __getitem__(self, index):
        image_feature=torch.from_numpy(self.data[index]["image_feature"])

        query_inputs = self.PQ_former_tokenizer(
            self.data[index]["query_input"],
            padding="max_length",
            truncation=True,
            max_length=self.num_query_token,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]

        scene_graph = self.PQ_former_tokenizer(
            self.data[index]["scene_graph"],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        scene_graph={
                "input_ids":scene_graph["input_ids"],
                "attention_mask":scene_graph["attention_mask"]
        }

        IE_inputs = self.IE_tokenizer(
            text=self.data[index]["query_input"],
            text_pair=self.data[index]["text_input"].replace(" ###",","),
            padding="max_length",
            truncation=True,
            max_length=(self.max_seq_len-self.num_query_token),
            add_special_tokens=True,
        )
        # pdb.set_trace()
        IE_inputs["input_ids"]=[self.SEP_token_id if x == self.split_token_id else x for x in IE_inputs["input_ids"]]
        
        IE_inputs["input_ids"]=torch.tensor(IE_inputs["input_ids"]).int()
        IE_inputs["attention_mask"]=torch.tensor(IE_inputs["attention_mask"]).int()
        IE_inputs={
                    "input_ids":IE_inputs["input_ids"],
                    "attention_mask":IE_inputs["attention_mask"]
        }

        # aspect task label
        aspect_exist_targets = torch.tensor(self.data[index]['aspect_exist_label']).int()

        # img text match_target
        match_target=torch.tensor(self.data[index]['match_label']).int()

        # task type
        task_type=self.data[index]['type']

        res=[image_feature, query_inputs, scene_graph, IE_inputs, aspect_exist_targets, match_target, task_type]
        
        return tuple(res)

    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
    #batch:[image_feature, query_inputs, answer_inputs, IE_inputs, start_ids, end_ids]
    image_embeds=torch.stack([b[0] for b in batch], dim=0)
    query_inputs=torch.stack([b[1] for b in batch], dim=0)
    answer_inputs={
                    "input_ids":torch.stack([b[2]["input_ids"][0] for b in batch], dim=0),
                    "attention_mask":torch.stack([b[2]["attention_mask"][0] for b in batch], dim=0)
                    }
    IE_inputs={
                "input_ids":torch.stack([b[3]["input_ids"] for b in batch], dim=0),
                "attention_mask":torch.stack([b[3]["attention_mask"] for b in batch], dim=0)
                    }
    aspect_exist_targets=torch.stack([b[4] for b in batch], dim=0)
    match_target=torch.stack([b[5] for b in batch], dim=0)
    task_type=[b[6] for b in batch]
    sample={"image_embeds":image_embeds,"query_inputs":query_inputs,
            "answer_inputs":answer_inputs,"IE_inputs":IE_inputs,
            "aspect_exist_targets":aspect_exist_targets,"match_target":match_target,
            "task_type":task_type}
    
    return sample

if __name__=="__main__":
    PQ_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    IE_tokenizer=BertTokenizer.from_pretrained('./Text_encoder/model_best')
    eval_ds= pretrain_dataset( 
                    data_path="/home/data/pretrain_dataset/test/processed_multitask_Pretrain_test_dataset.pkl",
                    max_seq_len=512,
                    IE_tokenizer=IE_tokenizer,
                    PQ_former_tokenizer=PQ_tokenizer,
                    num_query_token=32,
                    SEP_token_id=2,
                    split_token_id=187284,
                    set_size=1)
    eval_ds.update_data()
    eval_dataloader=DataLoader(eval_ds,batch_size =3,collate_fn=collate_fn,shuffle=False)
    batch=next(iter(eval_dataloader))
    input_id=batch["IE_inputs"]["input_ids"][1]
    prompt_mask=batch["prompt_mask"][1]
    print(input_id)
    print(prompt_mask)
    pdb.set_trace()
