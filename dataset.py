import numpy as np
import torch
import sys
sys.path.append('Text_encoder')
sys.path.append('PDQ')
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer
import pdb
from tqdm import tqdm
from PIL import Image
import os
import os.path as osp
import pickle
import random
import json

def get_tgt(target,tokenizer):
    tgt_list = [tokenizer.encode(x, add_special_tokens=False) for x in target]
    return tgt_list

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
            start_pos.append(i)
            end_pos.append(i + tgt_token_len - 1)
    return start_pos,end_pos

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if hasattr(self,'_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if hasattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_):
    """
    Convert a sequence of head indexes into a tree object.
    """
    if isinstance(head, list) == False:
        tokens = tokens[:len_].tolist()
        head = head[:len_].tolist()
    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1 # just a filler
        if h == 0:
            root = nodes[i]
        else:
            try:
                nodes[h-1].add_child(nodes[i])
            except:
                print(len_)
                exit()

    # 如果没有找到根节点，可以手动创建一个或选择第一个节点作为根
    if root is None:
        # print("Warning: No root node (head=0) found in the dependency tree.")
        # 选项1：使用第一个节点作为根
        # root = nodes[0]
        # 选项2：创建一个新的根节点并将所有没有父节点的节点连接到它
        root = Tree()
        root.idx = len(nodes)
        for node in nodes:
            if node.parent is None and node != root:
                root.add_child(node)

    assert root is not None
    return root

def tree_to_adj(sent_len, tree, directed=False, self_loop=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            if t.idx >= sent_len and c.idx >= sent_len:
                ret[sent_len-1,sent_len-1] = 1
            elif t.idx >= sent_len:
                ret[sent_len-1, c.idx] = 1
            elif c.idx >= sent_len:
                # print(t.idx)
                # print(sent_len)
                ret[t.idx, sent_len-1] = 1
            else:
                # print(t.idx)
                # print(sent_len)
                ret[t.idx, c.idx] = 1
            
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            if i >=sent_len:
                ret[sent_len-1, sent_len-1] = 1
            else:
                ret[i, i] = 1

    return ret

def inputs_to_tree_reps(maxlen, head, words, l):
    tree = head_to_tree(head, words, l)  # 将依存句法解析结果转换为树结构
    adj = tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(maxlen, maxlen)  # 生成无向图邻接矩阵，含自环（self_loop）   
    return adj

# 解析包含情感分析标注的JSON文件
def ParseData(data, max_seq_len, left_len):
    polar_dict = {'POS':2, 'NEG':0, 'NEU':-1}  # 情感标签映射

    d = data['parse_info']
    text_list = list(d['token'])

    tok = d['token']                # word token
    length = len(tok)               # real length
    tok = [t.lower() for t in tok]
    tok = ' '.join(tok)
    
    pos = d['postag']               # pos_tag 词性标注
    head = d['edges']               # head 
    head = [int(x) for x in head]   # 依存头索引
    deprel = d['deprels']           # deprel  依存关系类型

    adj = inputs_to_tree_reps(length, head, tok, length)
    assert len(text_list) == adj.shape[0] == adj.shape[1], '{}-{}-{}'.format(len(text_list), text_list, adj.shape)
    ori_adj_matrix = np.zeros((max_seq_len, max_seq_len)).astype('float32')
    ori_adj_matrix[left_len : left_len+length, left_len : left_len+length] = adj

    aspects_item = []
    for aspect in d['aspects']:
        asp = str(aspect['term']).lower()

        aspect_post = [int(aspect['from']), int(aspect['to'])]
        # label = polar_dict[aspect['polarity']]
        label = 1  # 此处有问题
        
        s_b,s_e = aspect['scope']
        aspect_scope = [0]*s_b + [1]*(s_e - s_b + 1) + [0]*(length - s_e -1)
        if len(aspect_scope) != length:
            if len(aspect_scope) < length:
                aspect_scope = aspect_scope + [0]*(length - len(aspect_scope))
            else:
                aspect_scope = aspect_scope[:length]

        assert len(aspect_scope) == length, f"Scope length {len(aspect_scope)} doesn't match expected length {length}"
        aspect_sample = {'aspect': asp, 'aspect_post': aspect_post, 'label': label, 'span_mask': aspect_scope}
        aspects_item.append(aspect_sample)
    
    nouns_item = []
    for noun in data['nouns']:
        term = noun['term']
        term = str(term).lower()
        noun_target = 1 if any(term == i['aspect'] for i in aspects_item) else 0

        term_post = [int(noun['from']), int(noun['to'])]
        id_b, id_e = term_post
        n_mask = [0]*id_b + [1]*(id_e - id_b + 1) + [0]*(length - id_e -1)
        
        if len(n_mask) != length:
            if len(n_mask) < length:
                n_mask = n_mask + [0]*(length- len(n_mask))
            else:
                n_mask = n_mask[:length]

        t_b,t_e = noun['scope']
        noun_scope = [0]*t_b + [1]*(t_e - t_b + 1) + [0]*(length - t_e -1)

        if len(noun_scope) != length:
            if len(noun_scope) < length:
                noun_scope = noun_scope + [0]*(length - len(noun_scope))
            else:
                noun_scope = noun_scope[:length]

        assert len(noun_scope) == length, f"Scope length {len(noun_scope)} doesn't match expected length {length}"
        noun_sample = {'term': term, 'term_post': term_post, 'noun_mask': n_mask, 'noun_target': noun_target, 'noun_scope': noun_scope}
        nouns_item.append(noun_sample)

    sample = {'text_list': text_list, 'text': tok, 'length': length, 'pos': pos, 'head': head, 'deprel': deprel, 
              'adj_matrix': ori_adj_matrix, 'aspects_item': aspects_item, 'nouns_item': nouns_item}

    return sample

def calculate_cls_sep_length(IE_inputs, tokenizer):
    # 获取特殊标记的 ID 
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    input_ids = IE_inputs["input_ids"].squeeze().tolist()
 
    try:
        cls_pos = input_ids.index(cls_token_id) 
        sep_pos = input_ids.index(sep_token_id) 
        return sep_pos - cls_pos + 1 
    except ValueError: 
        return 0 

class twitter_dataset(Dataset):
    def __init__(self,
                IE_tokenizer,
                PQ_former_tokenizer,
                data_path,                
                max_seq_len=512,
                num_query_token=32,
                SEP_token_id=2,
                split_token_id=187284,
                set_size=10,
                with_label=False,
                with_prompt_mask=True
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
        self.with_label=with_label
        self.with_prompt_mask=with_prompt_mask
        if with_label:
            self.label_data=[]
            for x in label_filelist:
                with open (x,"r")as f:
                    temp=json.load(f)
                    self.label_data.extend(temp)
        else:
            self.label_data=None

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
            text=self.data[index]["query_input"],  # prompt + text_input
            text_pair=self.data[index]["text_input"].replace(" ###",","),
            padding="max_length",
            truncation=True,
            max_length=(self.max_seq_len-self.num_query_token),
            add_special_tokens=True,
            #return_offsets_mapping=True
        )
        # pdb.set_trace()
        IE_inputs["input_ids"]=[self.SEP_token_id if x == self.split_token_id else x for x in IE_inputs["input_ids"]]
        
        IE_inputs["input_ids"]=torch.tensor(IE_inputs["input_ids"]).int()
        IE_inputs["attention_mask"]=torch.tensor(IE_inputs["attention_mask"]).int()
        IE_inputs={
                    "input_ids":IE_inputs["input_ids"],
                    "attention_mask":IE_inputs["attention_mask"]
        }

        # start_ids and end_ids
        start_ids = torch.zeros(self.max_seq_len).int()
        end_ids = torch.zeros(self.max_seq_len).int()
        if isinstance(self.data[index]["target"],list):
            for i in self.data[index]["target"]:
                start_pos_list,end_pos_list=get_span(target=i,
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=self.IE_tokenizer)
                for j in range(len(start_pos_list)):
                    if (not start_ids[start_pos_list[j]+self.num_query_token]==1) and (not end_ids[end_pos_list[j]+self.num_query_token]==1):
                        start_ids[start_pos_list[j]+self.num_query_token]=1
                        end_ids[end_pos_list[j]+self.num_query_token] = 1
        else:
            start_pos_list, end_pos_list=get_span(self.data[index]["target"],
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=self.IE_tokenizer)
            for start_pos in start_pos_list:
                start_ids[start_pos+self.num_query_token] = 1
            for end_pos in end_pos_list:
                end_ids[end_pos+self.num_query_token] = 1

        # aspect mask for each aspect
        aspects_mask = []
        if isinstance(self.data[index]["target"],list):
            for i in self.data[index]["target"]:
                aspect_mask = torch.zeros(self.max_seq_len).int()
                start_pos_list,end_pos_list=get_span(target=i,
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=self.IE_tokenizer)
                for j in range(len(start_pos_list)):
                    if (not aspect_mask[start_pos_list[j]+self.num_query_token]==1) and (not aspect_mask[end_pos_list[j]+self.num_query_token]==1):
                        aspect_mask[start_pos_list[j]+self.num_query_token:end_pos_list[j]+self.num_query_token+1]=1
                aspects_mask.append(aspect_mask)
        else:
            aspect_mask = torch.zeros(self.max_seq_len).int()
            start_pos_list,end_pos_list=get_span(self.data[index]["target"],
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=self.IE_tokenizer)
            for start_pos, end_pos in zip(start_pos_list, end_pos_list): 
                aspect_mask[start_pos+self.num_query_token:end_pos+self.num_query_token+1]=1 
            aspects_mask.append(aspect_mask)

        prompt_length_stoken = calculate_cls_sep_length(IE_inputs, self.IE_tokenizer)
        left_tokens_len = self.num_query_token + prompt_length_stoken
        parse_info = ParseData(self.data[index], self.max_seq_len, left_tokens_len)
        
        # 此处有问题，一个item有多个aspect scope和多个noun scope，但现在只假定有一个
        # aspect scope mask for each aspect
        aspects_scope = []
        for aspect_item in parse_info['aspects_item']:
            span = aspect_item['span_mask']
            span_mask = torch.zeros(self.max_seq_len).int()
            span_mask[left_tokens_len : parse_info['length'] + left_tokens_len] = torch.tensor(span, dtype=torch.int) 
            aspects_scope.append(span_mask) 
        aspects_scope = torch.stack(aspects_scope, dim=0)

        # nouns mask for each noun
        nouns_mask = []
        for noun_item in parse_info['nouns_item']:
            n_mask = noun_item['noun_mask']
            noun_mask = torch.zeros(self.max_seq_len).int()
            noun_mask[left_tokens_len : parse_info['length'] + left_tokens_len]= torch.tensor(n_mask, dtype=torch.int)
            # 下面这种写法和noun scope对不上，按道理noun scope要包含noun mask
            # start_pos_list,end_pos_list=get_span(target=noun_item['term'],
            #                             input_ids=IE_inputs["input_ids"],
            #                             tokenizer=self.IE_tokenizer)
            # for start_pos, end_pos in zip(start_pos_list, end_pos_list): 
            #     noun_mask[start_pos+self.num_query_token:end_pos+self.num_query_token+1]=1
            nouns_mask.append(noun_mask)
        nouns_mask = torch.stack(nouns_mask, dim=0)

        # nouns scope mask for each noun
        nouns_scope = []
        for noun_item in parse_info['nouns_item']:
            n_scope = noun_item['noun_scope']
            noun_span_mask = torch.zeros(self.max_seq_len).int()
            noun_span_mask[left_tokens_len : parse_info['length'] + left_tokens_len] = torch.tensor(n_scope, dtype=torch.int)
            nouns_scope.append(noun_span_mask)
        nouns_scope = torch.stack(nouns_scope, dim=0)

        # adj_matrix
        adj_matrix=torch.tensor(parse_info["adj_matrix"])

        # noun_term_target
        noun_targets = []
        for noun_item in parse_info['nouns_item']:
            n_target = noun_item['noun_target']
            noun_targets.append(n_target)
        noun_targets = torch.tensor(noun_targets)

        res=[image_feature, query_inputs, scene_graph, IE_inputs, start_ids, end_ids, aspects_mask, aspects_scope, nouns_mask, nouns_scope, adj_matrix, noun_targets]
        # res=[image_feature, query_inputs, answer_inputs, IE_inputs, start_ids, end_ids]

        # label
        if self.with_label:
            res.append(self.label_data[index])
        else:
            res.append(None)
        
        # prompt mask  将MATE和MASC都看作是词提取任务
        if self.with_prompt_mask:
            if "[ positive, neutral, negative ]" in self.data[index]["query_input"]:
                prompt_target="[ positive, neutral, negative ]"
            else:
                prompt_target=self.data[index]["text_input"]

            prompt_mask = torch.zeros(self.max_seq_len,self.max_seq_len).int()
            prompt_start_list,prompt_end_list=get_span(prompt_target,
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=self.IE_tokenizer)
            for start_pos,end_pos in zip(prompt_start_list,prompt_end_list):
                prompt_mask[start_pos+self.num_query_token:end_pos+self.num_query_token+1,
                            start_pos+self.num_query_token:end_pos+self.num_query_token+1]=1
            res.append(prompt_mask)
        else:
            res.append(None)

        return tuple(res)

    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
    #batch:[image_feature, query_inputs, answer_inputs, IE_inputs, start_ids, end_ids, ...]
    image_embeds=torch.stack([b[0] for b in batch], dim=0)
    query_inputs=torch.stack([b[1] for b in batch], dim=0)
    scene_graph={
                    "input_ids":torch.stack([b[2]["input_ids"][0] for b in batch], dim=0),
                    "attention_mask":torch.stack([b[2]["attention_mask"][0] for b in batch], dim=0)
                    }
    IE_inputs={
                "input_ids":torch.stack([b[3]["input_ids"] for b in batch], dim=0),
                "attention_mask":torch.stack([b[3]["attention_mask"] for b in batch], dim=0)
                    }
    start_ids=torch.stack([b[4] for b in batch], dim=0)
    end_ids=torch.stack([b[5] for b in batch], dim=0)

    aspects_mask=[b[6] for b in batch]
    aspects_scope=[b[7] for b in batch]
    nouns_mask=[b[8] for b in batch]
    nouns_scope=[b[9] for b in batch]

    # aspects_mask=torch.stack([b[6] for b in batch], dim=0)
    # aspects_scope=torch.stack([b[7] for b in batch], dim=0)
    # nouns_mask=torch.stack([b[8] for b in batch], dim=0)
    # nouns_scope=torch.stack([b[9] for b in batch], dim=0)

    adj_matrix=torch.stack([b[10] for b in batch], dim=0)
    noun_targets=[b[11] for b in batch]
    # aspect_targets=torch.stack([b[11] for b in batch], dim=0)

    sample={"image_embeds":image_embeds,"query_inputs":query_inputs,"scene_graph":scene_graph,"IE_inputs":IE_inputs,
            "start_ids":start_ids,"end_ids":end_ids,"aspects_mask":aspects_mask,"aspects_scope":aspects_scope,
            "nouns_mask":nouns_mask,"nouns_scope":nouns_scope,"adj_matrix":adj_matrix,"noun_targets":noun_targets
            }
    
    try:
        if batch[0][12]!=None:
            label_data=[x[12] for x in batch ]
            sample["label_data"]=label_data
    except:
        pdb.set_trace()

    if batch[0][13]!=None:
        prompt_mask=torch.stack([b[13] for b in batch], dim=0)
        sample["prompt_mask"]=prompt_mask
    
    return sample