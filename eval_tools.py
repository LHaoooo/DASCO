import torch
import argparse
import contextlib
import torch
from dataset import collate_fn, twitter_dataset
from tqdm import tqdm

def compute_metric(total_correct,total_label,total_pred):
    precision = total_correct / total_pred if total_correct else 0.0
    recall=total_correct/total_label if total_correct else 0.0
    f1=(2 * (precision * recall) / (precision + recall)) if total_correct else 0.0
    return precision,recall,f1

def compute_metric_macro(total_correct,total_label,merged=None):
    classes = [0, 1, 2]
    Accuracy=total_correct/total_label if total_label else 0.0

    # 计算macro F1 
    f1_scores = []
    for cls in classes:
        tp = merged[cls]['tp'].item()
        fp = merged[cls]['fp'].item()
        fn = merged[cls]['fn'].item()
 
        # 处理除零保护 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0 
        
        f1_scores.append(f1) 
    
    macro_f1 = sum(f1_scores) / len(f1_scores)
    
    return Accuracy, macro_f1

def eval_MATE(model,dataloader,device='cpu'):
    
    model.to(device)
    model.eval()
    total_correct = 0
    total_label = 0
    total_pred = 0

    with torch.no_grad():
        for batch in tqdm(dataloader,desc="evaluating model"):
            batch["image_embeds"]=batch["image_embeds"].to(device)
            batch["query_inputs"] = batch["query_inputs"].to(device)
            batch["scene_graph"]['input_ids'] = batch["scene_graph"]['input_ids'].to(device)  # [128, 512]
            batch["scene_graph"]['attention_mask'] = batch["scene_graph"]['attention_mask'].to(device)  # [128, 512]
            batch["IE_inputs"]['input_ids'] = batch["IE_inputs"]['input_ids'].to(device)
            batch["IE_inputs"]['attention_mask'] = batch["IE_inputs"]['attention_mask'].to(device)
            batch["adj_matrix"]=batch["adj_matrix"].to(device)

            with maybe_autocast(model):
                with torch.no_grad():
                    output = model(batch,no_its_and_itm=True)

            total_correct += output.n_correct
            total_pred += output.n_pred
            total_label += output.n_label
    
    model.train()
    return torch.tensor(total_correct).to(device),torch.tensor(total_label).to(device),torch.tensor(total_pred).to(device)

def eval_MASC(model,dataloader,device='cpu'):
    
    model.to(device)
    model.eval()
    total_correct = 0
    total_label = 0
    total_pred = 0
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
            batch["adj_matrix"]=batch["adj_matrix"].to(device)

            with maybe_autocast(model):
                output = model(batch,no_its_and_itm=True)
            
            total_correct += output.n_correct
            total_pred += output.n_pred
            total_label += output.n_label
            for cls in classes:
                merged[cls]['tp'] += output.class_stats[cls]['tp']
                merged[cls]['fp'] += output.class_stats[cls]['fp']
                merged[cls]['fn'] += output.class_stats[cls]['fn']
    
    for cls in classes:
        merged[cls]['tp'] = torch.tensor(merged[cls]['tp'], device=device)
        merged[cls]['fp'] = torch.tensor(merged[cls]['fp'], device=device)
        merged[cls]['fn'] = torch.tensor(merged[cls]['fn'], device=device)
            
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
    
def eval_MABSA(MATE_model,MASC_model,dataloader,device='cpu'):
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
        batch["adj_matrix"]=batch["adj_matrix"].to(device)

        with maybe_autocast(MATE_model):
            with torch.no_grad():
                output = MATE_model(batch,no_its_and_itm=True)
                # print(output.n_correct,output.n_pred,output.n_label)
                new_batch = output.new_batch
                false_batch = output.false_batch
        with maybe_autocast(MASC_model):
            with torch.no_grad():      
                masc_output = MASC_model(new_batch,no_its_and_itm=True)
                false_output = MASC_model(false_batch,no_its_and_itm=True)

        total_correct += (masc_output.n_correct - false_output.n_correct)
        total_pred += output.n_pred
        total_label += output.n_label

    return torch.tensor(total_correct).to(device),\
           torch.tensor(total_label).to(device),\
           torch.tensor(total_pred).to(device)

if __name__=="__main__":
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from transformers import BertTokenizer
    from tqdm import tqdm
    from model import from_pretrained
    from dataset import collate_fn, twitter_dataset
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument('--MATE_model', type=str, default=None)
    parser.add_argument('--MASC_model', type=str, default=None)
    parser.add_argument('--test_ds', type=str, default="./playground/twitter2015/MASC/test")
    parser.add_argument('--base_model', type=str, default="./Text_encoder/model_best")
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--hyper1', type=float, default=0.2)
    parser.add_argument('--hyper2', type=float, default=0.2)
    parser.add_argument('--hyper3', type=float, default=0.2)
    parser.add_argument('--gcn_layers', type=int, default=4)

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
    eval_dataloader = DataLoader(eval_ds, batch_size=128, collate_fn=collate_fn, shuffle=False)

    device=args.device

    if args.task=="MATE" :
        model = from_pretrained(args.MATE_model, args)
        c, l, p = eval_MATE(model, eval_dataloader, device=device)
        a, r, f1 = compute_metric(c, l, p)
        print(f"Correct:{c}, Label:{l}, Prediction:{p}; Accuracy:{100 * a:.3f}, Recall:{100 * r:.3f}, F1:{100 * f1:.3f}")

    if args.task=="MASC" :
        model = from_pretrained(args.MASC_model, args)
        c,l,p,merged=eval_MASC(model,eval_dataloader,device=device)
        a, f1 = compute_metric_macro(c, l, merged)
        print(f"Correct:{c}, Label:{l}, Prediction:{p}; Accuracy:{100 * a:.3f}, Macro_f1:{100 * f1:.3f}")

    if args.task== "MABSA":
        MATE_model = from_pretrained(args.MATE_model, args)
        args.task= "MASC"
        MASC_model = from_pretrained(args.MASC_model, args)
        c, l, p = eval_MABSA(MATE_model, MASC_model, eval_dataloader, device=device)
        a, r, f1 = compute_metric(c, l, p)
        print(f"Correct:{c}, Label:{l}, Prediction:{p}; Accuracy:{100 * a:.3f}, Recall:{100 * r:.3f}, F1:{100 * f1:.3f}")