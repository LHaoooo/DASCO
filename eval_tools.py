import torch
import argparse
import contextlib
import torch
from dataset import collate_fn, twitter_dataset,get_span
from tqdm import tqdm

def get_span_for_eval_2d(span_pred):
    span_list=torch.nonzero(span_pred)
    res=[]
    for span in span_list:
        tmp=[span[0],span[1]]
        res.append(tmp)
    return res

def tokens2text(tokenizer,pred_tokens):
    pred_entities=[tokenizer.decode(x) for x in pred_tokens]
    for x in pred_entities:
        x=x.replace("[PAD]","")
    return pred_entities

def compute_metric(total_correct,total_label,total_pred):
    precision = total_correct / total_pred if total_correct else 0.0
    recall=total_correct/total_label if total_correct else 0.0
    f1=(2 * (precision * recall) / (precision + recall)) if total_correct else 0.0
    return precision,recall,f1

def compute_metric_macro(total_correct,total_label,merged=None):
    classes = [0, 1, 2]
    Accuracy=total_correct/total_label if total_correct else 0.0

    # 计算macro F1 
    f1_scores = []
    for cls in classes:
        tp = merged[cls]['tp'].sum().item()
        fp = merged[cls]['fp'].sum().item()
        fn = merged[cls]['fn'].sum().item()
 
        # 处理除零保护 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0 
        
        f1_scores.append(f1) 
    
    macro_f1 = sum(f1_scores) / len(f1_scores)
    
    return Accuracy, macro_f1

def eval_MATE(model,dataloader,limit=0.5,device='cpu'):
    
    model.to(device)
    model.eval()
    total_correct = 0
    total_label = 0
    total_pred=0

    with torch.no_grad():
        for batch in tqdm(dataloader,desc="evaluating model"):
            batch["image_embeds"]=batch["image_embeds"].to(device)
            batch["query_inputs"] = batch["query_inputs"].to(device)
            batch["scene_graph"]['input_ids'] = batch["scene_graph"]['input_ids'].to(device)  # [128, 512]
            batch["scene_graph"]['attention_mask'] = batch["scene_graph"]['attention_mask'].to(device)  # [128, 512]
            batch["IE_inputs"]['input_ids'] = batch["IE_inputs"]['input_ids'].to(device)
            batch["IE_inputs"]['attention_mask'] = batch["IE_inputs"]['attention_mask'].to(device)
            batch["start_ids"]=batch["start_ids"].to(device)
            batch["end_ids"]=batch["end_ids"].to(device)
            batch["adj_matrix"]=batch["adj_matrix"].to(device)
            batch["prompt_mask"]=batch["prompt_mask"].to(device)

            with maybe_autocast(model):
                with torch.no_grad():
                    output = model(batch,no_its_and_itm=True)

            total_correct += output.n_correct
            total_pred += output.n_pred
            total_label += output.n_label
    
    model.train()
    return torch.tensor(total_correct).to(device),torch.tensor(total_label).to(device),torch.tensor(total_pred).to(device)

def eval_MASC(model,dataloader,limit=0.5,device='cpu'):
    
    IE_tokenizer=model.tokenizer
    model.to(device)
    model.eval()
    total_correct = 0
    total_label = 0
    total_pred=0
    classes = [0, 1, 2]
    merged = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in classes}

    with torch.no_grad():
        for batch in tqdm(dataloader,desc="evaluating model"):
            batch["image_embeds"]=batch["image_embeds"].to(device)
            batch["query_inputs"] = batch["query_inputs"].to(device)
            batch["scene_graph"]['input_ids'] = batch["scene_graph"]['input_ids'].to(device)  # [128, 512]
            batch["scene_graph"]['attention_mask'] = batch["scene_graph"]['attention_mask'].to(device)  # [128, 512]
            batch["IE_inputs"]['input_ids'] = batch["IE_inputs"]['input_ids'].to(device)
            batch["IE_inputs"]['attention_mask'] = batch["IE_inputs"]['attention_mask'].to(device)
            batch["start_ids"]=batch["start_ids"].to(device)
            batch["end_ids"]=batch["end_ids"].to(device)
            batch["adj_matrix"]=batch["adj_matrix"].to(device)
            batch["prompt_mask"]=batch["prompt_mask"].to(device)

            with maybe_autocast(model):
                with torch.no_grad():
                    output = model(batch,no_its_and_itm=True)
            
            total_correct += output.n_correct
            total_pred += output.n_pred
            total_label += output.n_label
            for cls in classes:
                merged[cls]['tp'] += output.class_stats[cls]['tp']
                merged[cls]['fp'] += output.class_stats[cls]['fp']
                merged[cls]['fn'] += output.class_stats[cls]['fn']
    
    for cls in classes:
        merged[cls]['tp'] = torch.tensor(merged[cls]['tp']).to(device)
        merged[cls]['fp'] = torch.tensor(merged[cls]['fp']).to(device)
        merged[cls]['fn'] = torch.tensor(merged[cls]['fn']).to(device)
            
    model.train()
    return torch.tensor(total_correct).to(device),torch.tensor(total_label).to(device),torch.tensor(total_pred).to(device), merged

def maybe_autocast(model, device=None,dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    if device is not None:
        enable_autocast = torch.device(device) != torch.device("cpu")
    else:
        enable_autocast = next(model.parameters()).device != torch.device("cpu")
    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()
    
def get_best_pred_span(span_pred):
    best_span_index=torch.argmax(span_pred)
    index_1d=best_span_index//span_pred.shape[-1]
    index_2d=best_span_index-index_1d*span_pred.shape[-1]
    best_pred_span=[[index_1d,index_2d]]
    return best_pred_span

def single_batch_pred(model,batch,device,limit=0.5,no_its_and_itm=False,return_best_pred=False):
    
    
    
    batch["image_embeds"]=batch["image_embeds"].to(device)
    batch["query_inputs"] = batch["query_inputs"].to(device)
    batch["answer_inputs"]['input_ids'] = batch["answer_inputs"]['input_ids'].to(device)
    batch["answer_inputs"]['attention_mask'] = batch["answer_inputs"]['attention_mask'].to(device)
    batch["IE_inputs"]['input_ids'] = batch["IE_inputs"]['input_ids'].to(device)
    batch["IE_inputs"]['attention_mask'] = batch["IE_inputs"]['attention_mask'].to(device)
    batch["start_ids"]=batch["start_ids"].to(device)
    batch["end_ids"]=batch["end_ids"].to(device)
    batch["prompt_mask"]=batch["prompt_mask"].to(device)

    with maybe_autocast(model):
        b=batch["image_embeds"].size()[0]
        output = model(batch,no_its_and_itm=no_its_and_itm)
        span_pred=output.span_prob.float()
        span_pred[span_pred<limit]=0
        texts=batch["IE_inputs"]['input_ids']


    IE_tokenizer=model.tokenizer

    pred_entities_list=[]
    for i in range(b):
        if not return_best_pred:
            pred_spans=get_span_for_eval_2d(span_pred[i])
        else:
            pred_spans=get_best_pred_span(span_pred[i])

        for x in range(len(pred_spans)):
            for p in range(2):
                pred_spans[x][p]=pred_spans[x][p]-32

        pred_tokens=get_split_tokens(texts[i],pred_spans)
        pred_entities=tokens2text(IE_tokenizer,pred_tokens)
        pred_entities_list.append(pred_entities)
    
    
    
    return pred_entities_list

def build_temp_samples(PQ_former_tokenizer,IE_tokenizer,image_embeds,pred_entities,text_input,prompt_mask,num_query_token=32,max_seq_len=512,sample_cls="sentiment",sentiment=None):
    #for sentiment: 
    temp_samples=[]
    image_feature=image_embeds
    for entity in pred_entities:
        if sample_cls=="sentiment":
            query_inputs_text=f'Sentiment of {entity} is [ positive, neutral, negative ]'
        elif sample_cls=="judgement":
            query_inputs_text=f'Sentiment of {entity} is {sentiment}, it is [ true, false ]'

        answer_inputs_text=f"It's a picture of {entity}"
        query_inputs = PQ_former_tokenizer(
                query_inputs_text,
                padding="max_length",
                truncation=True,
                max_length=num_query_token,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0]
        
        answer_inputs = PQ_former_tokenizer(
            answer_inputs_text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        )
        answer_inputs={
                "input_ids":answer_inputs["input_ids"],
                "attention_mask":answer_inputs["attention_mask"]
        }

        IE_inputs = IE_tokenizer(
            text=query_inputs_text,
            text_pair=text_input,
            padding="max_length",
            truncation=True,
            max_length=(max_seq_len-num_query_token),
            add_special_tokens=True,
            return_offsets_mapping=True
        )
        IE_inputs["input_ids"]=torch.tensor(IE_inputs["input_ids"]).int()
        IE_inputs["attention_mask"]=torch.tensor(IE_inputs["attention_mask"]).int()
        IE_inputs={
                    "input_ids":IE_inputs["input_ids"],
                    "attention_mask":IE_inputs["attention_mask"]
        }
        start_ids = torch.zeros(max_seq_len).int()
        end_ids = torch.zeros(max_seq_len).int()

        if isinstance(entity,list):
            for i in entity:
                start_pos_list,end_pos_list=get_span(target=i,
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=IE_tokenizer)
                for i in range(len(start_pos_list)):
                    if (not start_ids[start_pos_list[i]+num_query_token]==1) and (not end_ids[end_pos_list[i]+num_query_token]==1):
                        start_ids[start_pos_list[i]+num_query_token]=1
                        end_ids[end_pos_list[i]+num_query_token] = 1
        res=[image_feature, query_inputs, answer_inputs, IE_inputs, start_ids, end_ids]

        res.append(None)

        if "[ positive, neutral, negative ]" in query_inputs_text:
            prompt_target="[ positive, neutral, negative ]"
            prompt_mask = torch.zeros(max_seq_len,max_seq_len).int()
            prompt_start_list,prompt_end_list=get_span(prompt_target,
                                            input_ids=IE_inputs["input_ids"],
                                            tokenizer=IE_tokenizer)
            for start_pos,end_pos in zip(prompt_start_list,prompt_end_list):
                prompt_mask[start_pos+num_query_token:end_pos+num_query_token+1,
                            start_pos+num_query_token:end_pos+num_query_token+1]=1
        else:
            pass


        res.append(prompt_mask)
        temp_samples.append(res)
    return collate_fn(temp_samples)

def get_split_tokens(token_ids,span_list):
    tokens=[]
    for span in span_list:
        token=token_ids[span[0]:span[1]+1]
        tokens.append(token)
    return tokens

def get_span_for_eval(start_list,end_list):
    text_span=[]
    if len(start_list)==0 or len(end_list)==0:
        return[]
    i=0
    j=0
    while i<len(start_list)and j<len(end_list):
        if start_list[i]<=end_list[j]:
            text_span.append([start_list[i],end_list[j]])
            i+=1
        else:
            j+=1
    return text_span

def eval_MABSA(MATE_model,MASC_model,dataloader,MATE_limit=0.5,MASC_limit=0.3,device='cpu'):
    total_pred=0
    total_label=0
    total_correct=0

    MATE_model.to(device)
    MATE_model.eval()
    MASC_model.to(device)
    MASC_model.eval()


    for batch in tqdm(dataloader,desc="evaluating model"):
        #MATE
        batch["image_embeds"]=batch["image_embeds"].to(device)
        batch["query_inputs"] = batch["query_inputs"].to(device)
        batch["scene_graph"]['input_ids'] = batch["scene_graph"]['input_ids'].to(device)  # [128, 512]
        batch["scene_graph"]['attention_mask'] = batch["scene_graph"]['attention_mask'].to(device)  # [128, 512]
        batch["IE_inputs"]['input_ids'] = batch["IE_inputs"]['input_ids'].to(device)
        batch["IE_inputs"]['attention_mask'] = batch["IE_inputs"]['attention_mask'].to(device)
        batch["start_ids"]=batch["start_ids"].to(device)
        batch["end_ids"]=batch["end_ids"].to(device)
        batch["adj_matrix"]=batch["adj_matrix"].to(device)
        batch["prompt_mask"]=batch["prompt_mask"].to(device)

        with maybe_autocast(MATE_model):
            with torch.no_grad():
                output = MATE_model(batch,no_its_and_itm=True)
                new_batch = output.new_batch
        with maybe_autocast(MASC_model):
            with torch.no_grad():      
                masc_output = MASC_model(new_batch,no_its_and_itm=True)

        total_correct += masc_output.n_correct - output.false_num
        total_pred += output.n_pred
        total_label += output.n_label

    return torch.tensor(total_correct).to(device),\
           torch.tensor(total_label).to(device),\
           torch.tensor(total_pred).to(device)

mode_list=["MATE","MASC","Dual_JMASA"]
mode=mode_list[1]

if __name__=="__main__":
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from transformers import BertTokenizer
    from tqdm import tqdm
    from model import from_pretrained
    from dataset import collate_fn, twitter_dataset
    from torch.utils.data import Dataset, DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument('--MATE_model', type=str, default=None)
    parser.add_argument('--MASC_model', type=str, default=None)
    parser.add_argument('--test_ds', type=str, default="./playground/twitter2015/MASC/test")
    parser.add_argument('--base_model', type=str, default="./Text_encoder/model_best")
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--limit', type=float, default=0.5)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--hyper1', type=float, default=0.2)
    parser.add_argument('--hyper2', type=float, default=0.12)
    parser.add_argument('--hyper3', type=float, default=0.2)
    parser.add_argument('--gcn_layers', type=int, default=3)

    args = parser.parse_args()
    IE_tokenizer = BertTokenizer.from_pretrained(args.base_model)
    PQ_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.task=="MATE" or args.task=="MASC" :
        eval_ds = twitter_dataset(
            data_path=args.test_ds,
            max_seq_len=512,
            IE_tokenizer=IE_tokenizer,
            PQ_former_tokenizer=PQ_tokenizer,
            num_query_token=32,
            SEP_token_id=2,
            split_token_id=187284,
            set_size=1,
            task=args.task)
    elif args.task=="MABSA" :
        eval_ds = twitter_dataset(
            data_path=args.test_ds,
            max_seq_len=512,
            IE_tokenizer=IE_tokenizer,
            PQ_former_tokenizer=PQ_tokenizer,
            num_query_token=32,
            SEP_token_id=2,
            split_token_id=187284,
            set_size=1,
            task=args.task)
        
    eval_ds.update_data()
    eval_dataloader = DataLoader(eval_ds, batch_size=64, collate_fn=collate_fn, shuffle=False)

    limit = args.limit
    device=args.device

    if args.task=="MATE" :
        import pdb
        model = from_pretrained(args.MATE_model, args)
        c, l, p = eval_MATE(model, eval_dataloader, limit=limit, device=device)
        a, r, f1 = compute_metric(c, l, p)
        print(f"Correct:{c}, Label:{l}, Prediction:{p}; Accuracy:{100 * a:.3f}, Recall:{100 * r:.3f}, F1:{100 * f1:.3f}")

    if args.task=="MASC" :
        model = from_pretrained(args.MASC_model, args)
        c,l,p,merged=eval_MASC(model,eval_dataloader,limit=limit,device=device)
        a, f1 = compute_metric_macro(c, l, merged)
        print(f"Correct:{c}, Label:{l}, Prediction:{p}; Accuracy:{100 * a:.3f}, Macro_f1:{100 * f1:.3f}")

    if args.task== "MABSA":
        MATE_model = from_pretrained(args.MATE_model, args)
        args.gcn_layers=4
        args.task= "MASC"
        MASC_model = from_pretrained(args.MASC_model, args)
        MASC_limit=MATE_limit=args.limit
        c, l, p = eval_MABSA(MATE_model, MASC_model, eval_dataloader, MASC_limit=MASC_limit,MATE_limit=MATE_limit, device=device)
        a, r, f1 = compute_metric(c, l, p)
        print(f"Correct:{c}, Label:{l}, Prediction:{p}; Accuracy:{100 * a:.3f}, Recall:{100 * r:.3f}, F1:{100 * f1:.3f}")


