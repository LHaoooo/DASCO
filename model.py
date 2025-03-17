import sys
sys.path.append('Text_encoder')
sys.path.append('PDQ')
from transformers import BertTokenizerFast
from Text_encoder.sparse_attn_model import Text_encoder
from PDQ.PDQ import PDQ
from transformers.utils import ModelOutput
from typing import Optional, Dict, Any
import torch.nn as nn
import torch
from dataclasses import dataclass
import copy
import math
import torch.nn.functional as F

@dataclass
class DASCO_Output(ModelOutput):
    total_loss: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    loss_cl: Optional[torch.FloatTensor] = None
    n_correct: Optional[torch.LongTensor] = None
    n_pred: Optional[torch.LongTensor] = None
    n_label: Optional[torch.LongTensor] = None
    class_stats: Optional[Dict[int, Dict[str, torch.LongTensor]]] = None
    new_batch: Optional[Any] = None
    false_batch: Optional[Any] = None

def build_tokenizer(tokenizer_path):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn

class DASCO(nn.Module):
    def __init__(self, args, MFSUIE_config):
        super().__init__()
        self.pdq = PDQ()
        self.text_encoder = Text_encoder.from_pretrained(MFSUIE_config['text_model']["model_path"])
        self.tokenizer = build_tokenizer(MFSUIE_config["text_model"]["tokenizer_path"])
        
        Qformer_hidden_size = MFSUIE_config["pdq"]["hidden_size"]
        text_hidden_size = MFSUIE_config["text_model"]["hidden_size"]
        self.hidden_size = text_hidden_size
        self.FSUIE_proj = nn.Linear(Qformer_hidden_size, text_hidden_size)
        self.itc_weight = MFSUIE_config["loss_weights"]["itc"]
        self.itm_weight = MFSUIE_config["loss_weights"]["itm"]
        self.lm_weight = MFSUIE_config["loss_weights"]["lm"]
        self.cl_weight = MFSUIE_config["loss_weights"]["cl"]
        self.dropout_layer = nn.Dropout()

        self.task = args.task
        self.hyper1 = args.hyper1
        self.hyper2 = args.hyper2
        self.hyper3 = args.hyper3
        self.layers = args.gcn_layers

        self.attention_heads = 1
        self.mem_dim = self.hidden_size // 2
        self.attn = MultiHeadAttention(self.attention_heads, self.hidden_size)
        self.layernorm = LayerNorm(self.hidden_size)
        self.pooled_drop = nn.Dropout(0.3)

        # 双路GCN
        self.depW = nn.ModuleList() # DepGCN依存句法GCN
        for layer in range(self.layers):
            input_dim = text_hidden_size if layer == 0 else self.mem_dim
            self.depW.append(nn.Linear(input_dim, self.mem_dim))
            
        self.semW = nn.ModuleList()  # SemGCN语义GCN
        for j in range(self.layers):
            input_dim = text_hidden_size if j == 0 else self.mem_dim
            self.semW.append(nn.Linear(input_dim, self.mem_dim))
        
        self.fc1 = torch.nn.Linear(self.hidden_size//2, 32)
        self.fc2 = torch.nn.Linear(32, self.hidden_size//2)
        self.fc3 = nn.Linear(self.hidden_size//2, self.hidden_size//2)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.sigmoid = nn.Sigmoid()
        if self.task == 'MATE' or self.task == 'MABSA':
            self.classifier = nn.Linear(self.hidden_size*2, 2)
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == 'MASC':
            self.classifier = nn.Linear(self.hidden_size*2, 3)
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.14,0.46,0.4]))  #  twitter15[0.11, 0.59, 0.3] twitter17: [0.14,0.46,0.4]

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):  # cal cosine simility
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=2)
        return torch.bmm(z1, z2.transpose(1,2))
    
    def scope_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, s_mask, a_mask):
        f = lambda x: torch.exp(x / self.hyper2)  # f: e^(f(z1,z2)/t)
        Ba,Seq,Dim = z1.shape
        
        #aspect-mask
        a_mask = a_mask.unsqueeze(-1) #[B,S,1]
        asp_m = a_mask.expand(Ba,Seq,Dim) #[B,S,D]
        a_z1 = z1 * asp_m
        m_between_sim = f(self.sim(a_z1, z2)) # f(ma_h1, h2)
        m_refl_sim = f(self.sim(a_z1, z1)) # f(ma_h1, h1)
        
        #span-mask
        s_mask = s_mask.unsqueeze(-1) #[B,S,1]
        span_m = s_mask.expand(Ba,Seq,Dim) #[B,S,D]
        s_z1 = z1 * span_m
        s_z2 = z2 * span_m    
        as_refl_sim = f(self.sim(a_z1, s_z1)) # f(ma_h1, ms_h1)
        as_between_sim = f(self.sim(a_z1, s_z2)) # f(ma_h1, ms_h2)

        # weighted f()
        weighted_between_sim = f(torch.mul(self.sim(a_z1, s_z2), self.sim(a_z1, s_z2).diagonal(dim1=-2,dim2=-1).unsqueeze(dim=-1)))

        #Scope-asisted MvGCL
        pos = as_between_sim.diagonal(dim1=-2,dim2=-1) + (as_refl_sim.sum(2) - as_refl_sim.diagonal(dim1=-2,dim2=-1)) + (weighted_between_sim.sum(2) - weighted_between_sim.diagonal(dim1=-2,dim2=-1)) # 3
        alle = m_refl_sim.sum(2) + m_between_sim.sum(2) - m_refl_sim.diagonal(dim1=-2,dim2=-1)
        cl_logit = pos / alle

        return -torch.log(cl_logit)

    def scope_semi_loss_list(self, z1: torch.Tensor, z2: torch.Tensor, s_mask_list, a_mask_list):
        f = lambda x: torch.exp(x / self.hyper2)  # f: e^(f(z1,z2)/t)
        Ba,Seq,Dim = z1.shape
        
        results = []
        # 对每个batch单独处理
        for b in range(Ba):
            # 获取当前batch的mask
            s_masks = s_mask_list[b]  # [N, S]
            a_masks = a_mask_list[b]  # [N, S]

            # 获取当前batch的表示
            z1_b = z1[b:b+1]  # [1, Seq, Dim]
            z2_b = z2[b:b+1]  # [1, Seq, Dim]

            batch_results = []
            for i in range(len(s_masks)):
                s_mask = s_masks[i:i+1]  # [1, S]
                a_mask = a_masks[i:i+1]  # [1, S]

                a_mask = a_mask.unsqueeze(-1)  # [1, S, 1]
                asp_m = a_mask.expand(1, Seq, Dim)  # [1, Seq, Dim]
                a_z1 = z1_b * asp_m
                m_between_sim = f(self.sim(a_z1, z2_b))  # f(ma_h1, h2)
                m_refl_sim = f(self.sim(a_z1, z1_b))  # f(ma_h1, h1)

                s_mask = s_mask.unsqueeze(-1)  # [1, S, 1]
                span_m = s_mask.expand(1, Seq, Dim)  # [1, Seq, Dim]
                s_z1 = z1_b * span_m
                s_z2 = z2_b * span_m
                as_refl_sim = f(self.sim(a_z1, s_z1))  # f(ma_h1, ms_h1)
                as_between_sim = f(self.sim(a_z1, s_z2))  # f(ma_h1, ms_h2)

                # weighted f()
                weighted_between_sim = f(torch.mul(self.sim(a_z1, s_z2), 
                                        self.sim(a_z1, s_z2).diagonal(dim1=-2, dim2=-1).unsqueeze(dim=-1)))
                
                # Scope-asisted MvGCL
                pos = as_between_sim.diagonal(dim1=-2, dim2=-1) + \
                    (as_refl_sim.sum(2) - as_refl_sim.diagonal(dim1=-2, dim2=-1)) + \
                    (weighted_between_sim.sum(2) - weighted_between_sim.diagonal(dim1=-2, dim2=-1))  # 3
                alle = m_refl_sim.sum(2) + m_between_sim.sum(2) - m_refl_sim.diagonal(dim1=-2, dim2=-1)
                cl_logit = pos / alle

                batch_results.append(-torch.log(cl_logit))
            
            batch_results = torch.stack(batch_results, dim=0).mean(dim=0)
            results.append(batch_results)
        results = torch.stack(results).squeeze(1)
        return results # [B, S]

    def mate_cl_loss(self, samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs):
        adj_ag = None

        '''
        通过循环遍历每个注意力头的邻接矩阵，如果 adj_ag 还未初始化（即 None),则将当前注意力头的邻接矩阵赋值给 adj_ag;
        否则,将当前邻接矩阵累加到 adj_ag 上。
        最后将 adj_ag 除以注意力头的数量，得到平均后的邻接矩阵。
        '''
        # Attention matrix
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i].clone()
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        # 去除邻接矩阵的对角线元素，并添加自环，最后与文本编码器的注意力矩阵进行元素相乘。
        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = text_encoder_atts.transpose(1, 2) * adj_ag
        
        H_l = gcn_inputs
        si = nn.Sigmoid()
        relu = nn.ReLU()
        for l in range(self.layers):
            # **********GCN*********
            AH_sem = adj_ag.bmm(H_l)
            I_sem = self.semW[l](AH_sem) #SemGCN
            AH_dep = samples['adj_matrix'].bmm(H_l)
            I_dep = self.depW[l](AH_dep) #depGCN

            g = si(I_dep)
            I_dep_g = self.hyper1 * g # [B, S, 768/2]
            I_com = torch.mul((1-I_dep_g),I_sem) + torch.mul(I_dep_g,I_dep) # adaptive fusion
            H_out = relu(self.fc3(I_com))
            # H_out = nn.LayerNorm(H_out.size(-1), device=H_out.device)(H_out)
            
            if l == 0:
                H_l = self.fc4(H_l)
            g_l = si(H_l)
            H_l = torch.mul(g_l, H_out) + torch.mul((1 - g_l),H_l)

        # span-masked graphCL
        h1 = self.projection(H_l)
        h2 = self.projection(I_sem)  #[B,s,D/2]
        if no_its_and_itm:
            loss_cl = 0
        else:
            l1 = self.scope_semi_loss_list(h1, h2, samples['nouns_scope'], samples['nouns_mask'])
            l2 = self.scope_semi_loss_list(h2, h1, samples['nouns_scope'], samples['nouns_mask'])  # B, Seq
            loss = (l1 + l2) * 0.5
            loss_avg = loss.mean(dim=1, keepdim=True)
            loss_cl = loss_avg.mean()

        loss_target = 0
        n_correct = 0
        n_pred = 0
        n_label = 0
        for i in range(len(samples['nouns_mask'])):
            asp_wn_ori = samples['nouns_mask'][i].sum(dim=-1).unsqueeze(-1).to(h1.device) # [N,1]
            asp_wn_ori = torch.clamp(asp_wn_ori, min=1.0)
            n_mask_ori = samples['nouns_mask'][i].unsqueeze(-1).repeat(1, 1, self.hidden_size // 2).to(h1.device)  # [N,S,D/2]
            
            # 目标: 使h1.i和h2.i变为[1,S,D/2]以便与[N,S,D/2]的n_mask进行广播
            h1_expanded = h1[i].unsqueeze(0)  # [1, S, D/2]
            h2_expanded = h2[i].unsqueeze(0)  # [1, S, D/2]
            # 现在进行乘法操作，结果将广播为[N,S,D/2]
            masked_h1 = h1_expanded * n_mask_ori  # [N, S, D/2]
            masked_h2 = h2_expanded * n_mask_ori  # [N, S, D/2]
            # 对序列维度求和
            summed_h1 = masked_h1.sum(dim=1)  # [N, D/2]   
            summed_h2 = masked_h2.sum(dim=1)  # [N, D/2]
            # 确保asp_wn_ori形状为[N,1]以便除法广播
            # 如果asp_wn_ori已经是[N,1]形状，可以直接使用
            outputs1 = summed_h1 / asp_wn_ori  # [N, D/2]
            outputs2 = summed_h2 / asp_wn_ori  # [N, D/2]

            # 合并三个输出
            final_outputs = torch.cat((outputs1, outputs2, pooled_output[i].repeat(outputs2.size(0), 1)), dim=-1)
            logits = self.classifier(final_outputs)  # [N, 2]
            loss_target += self.criterion(logits, samples['noun_targets'][i].to(h1.device))
            
            labels = samples['noun_targets'][i].to(h1.device)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            n_correct += torch.sum(torch.logical_and(predictions == labels, labels == 1)).item()
            n_pred += torch.sum(predictions == 1).item()
            # n_label += torch.sum(labels == 1).item()
            n_label += samples['aspects_mask'][i].size(0)
        
        if no_its_and_itm:
            loss_cls_cl = 0
        else:
            loss_classify = loss_target.mean()
            loss_cls_cl = loss_classify +  self.hyper3 * loss_cl
        class_stats = None
        return loss_cls_cl, n_correct, n_pred, n_label, class_stats
    
    def mabsa_mate(self, samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs):
        adj_ag = None
        # Attention matrix
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = text_encoder_atts.transpose(1, 2) * adj_ag
        
        H_l = gcn_inputs
        for l in range(self.layers):
            si = nn.Sigmoid()
            # **********GCN*********
            AH_sem = adj_ag.bmm(H_l)
            I_sem = self.semW[l](AH_sem) #SemGCN
            AH_dep = samples['adj_matrix'].bmm(H_l)
            I_dep = self.depW[l](AH_dep) #depGCN

            g = si(I_dep)
            I_dep_g = self.hyper1 * g # [B, S, 768/2]
            I_com = torch.mul((1-I_dep_g),I_sem) + torch.mul(I_dep_g,I_dep) # adaptive fusion
            relu = nn.ReLU()
            H_out = relu(self.fc3(I_com))
            
            if l == 0:
                H_l = self.fc4(H_l)
            g_l = si(H_l)
            H_l = torch.mul(g_l, H_out) + torch.mul((1 - g_l),H_l)

        # span-masked graphCL
        h1 = self.projection(H_l)
        h2 = self.projection(I_sem)  #[B,s,D/2]

        n_correct = 0
        n_pred = 0
        n_label = 0
        res = []
        false_res = []
        for i in range(len(samples['nouns_mask'])):  # B
            asp_wn_ori = samples['nouns_mask'][i].sum(dim=-1).unsqueeze(-1).to(h1.device) # [N,1]
            asp_wn_ori = torch.clamp(asp_wn_ori, min=1.0)
            n_mask_ori = samples['nouns_mask'][i].unsqueeze(-1).repeat(1, 1, self.hidden_size // 2).to(h1.device)  # [N,S,D/2]
            h1_expanded = h1[i].unsqueeze(0)  # [1, S, D/2]
            h2_expanded = h2[i].unsqueeze(0)  # [1, S, D/2]
            masked_h1 = h1_expanded * n_mask_ori  # [N, S, D/2]
            masked_h2 = h2_expanded * n_mask_ori  # [N, S, D/2]
            summed_h1 = masked_h1.sum(dim=1)  # [N, D/2]   
            summed_h2 = masked_h2.sum(dim=1)  # [N, D/2]
            outputs1 = summed_h1 / asp_wn_ori  # [N, D/2]
            outputs2 = summed_h2 / asp_wn_ori  # [N, D/2]
            final_outputs = torch.cat((outputs1, outputs2, pooled_output[i].repeat(outputs2.size(0), 1)), dim=-1)
            logits = self.classifier(final_outputs)  # [N, 2]
            
            labels = samples['noun_targets'][i].to(h1.device)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            n_correct += torch.sum(torch.logical_and(predictions == labels, labels == 1)).item()
            n_pred += torch.sum(predictions == 1).item()
            n_label += samples['aspects_mask'][i].size(0)

            '''
            此处逻辑：
            先将所有预测的aspects_mask、aspects_scope、aspect_targets都放入一个列表1 new_batch_aspects中  然后masc
            不过此时会有错误预测的aspect作为masc的输入
            所以要将所有错误预测的aspect全放入另一个列表2 false_batch_aspects  对他们再进行masc
            最后真正的correct会是列表1中预测正确数 - 列表2中预测正确数
            pred则是列表1所有预测出的aspect数目
            label则无变化 是所有aspect的总数
            '''
            new_batch_aspects_mask = []
            new_batch_aspects_scope = []
            new_batch_aspect_targets = []

            false_batch_aspects_mask = []
            false_batch_aspects_scope = []
            false_batch_aspect_targets = []

            for j in range(len(predictions)):  # N
                if predictions[j] == 1 and labels[j] == 1:
                    for k in range(len(samples['aspects_mask'][i])):  # aspect num = k != j
                        if torch.all(samples['aspects_mask'][i][k] == samples['nouns_mask'][i][j]):
                            new_batch_aspects_mask.append(samples['aspects_mask'][i][k])
                            try:
                                new_batch_aspects_scope.append(samples['aspects_scope'][i][k])
                            except:
                                new_batch_aspects_scope.append(samples['aspects_mask'][i][k])
                            new_batch_aspect_targets.append(samples['aspect_targets'][i][k])
                elif predictions[j] == 1 and labels[j] == 0:
                    new_batch_aspects_mask.append(samples['nouns_mask'][i][j])
                    try:
                        new_batch_aspects_scope.append(samples['nouns_scope'][i][j])
                    except:
                        new_batch_aspects_scope.append(samples['nouns_mask'][i][j])
                    new_batch_aspect_targets.append(1)  # 错误预测的aspect的target 随便设，最后都要减去

                    false_batch_aspects_mask.append(samples['nouns_mask'][i][j])
                    try:
                        false_batch_aspects_scope.append(samples['nouns_scope'][i][j])
                    except:
                        false_batch_aspects_scope.append(samples['nouns_mask'][i][j])
                    false_batch_aspect_targets.append(1)
            
            
            # new batch 1
            if new_batch_aspects_mask == []:
                continue
            new_batch_aspects_mask = torch.stack(new_batch_aspects_mask, dim=0)
            new_batch_aspects_scope = torch.stack(new_batch_aspects_scope, dim=0)
            new_batch_aspect_targets = torch.tensor(new_batch_aspect_targets, device=h1.device)

            new_batch_sgids = samples['scene_graph']['input_ids'][i]
            new_batch_sg_attention_mask = samples['scene_graph']['attention_mask'][i]
            scene_captioning = {
                'input_ids': new_batch_sgids,
                'attention_mask': new_batch_sg_attention_mask
            }

            new_batch_inputids = samples['IE_inputs']['input_ids'][i]
            new_batch_attention_mask = samples['IE_inputs']['attention_mask'][i]
            text_input = {
                'input_ids': new_batch_inputids,
                'attention_mask': new_batch_attention_mask
            }
            
            res.append([samples['image_embeds'][i], samples['query_inputs'][i], scene_captioning, text_input,
                   new_batch_aspects_mask, new_batch_aspects_scope, samples['adj_matrix'][i], new_batch_aspect_targets])
            
            # false batch 2
            if false_batch_aspects_mask == []:
                continue
            false_batch_aspects_mask = torch.stack(false_batch_aspects_mask, dim=0)
            false_batch_aspects_scope = torch.stack(false_batch_aspects_scope, dim=0)
            false_batch_aspect_targets = torch.tensor(false_batch_aspect_targets, device=h1.device)

            false_batch_sgids = samples['scene_graph']['input_ids'][i]
            false_batch_sg_attention_mask = samples['scene_graph']['attention_mask'][i]
            false_scene_captioning = {
                'input_ids': false_batch_sgids,
                'attention_mask': false_batch_sg_attention_mask
            }

            false_batch_inputids = samples['IE_inputs']['input_ids'][i]
            false_batch_attention_mask = samples['IE_inputs']['attention_mask'][i]
            false_text_input = {
                'input_ids': false_batch_inputids,
                'attention_mask': false_batch_attention_mask
            }
            
            false_res.append([samples['image_embeds'][i], samples['query_inputs'][i], false_scene_captioning, false_text_input,
                   false_batch_aspects_mask, false_batch_aspects_scope, samples['adj_matrix'][i], false_batch_aspect_targets])
        
        image_embeds=torch.stack([b[0] for b in res], dim=0)
        query_inputs=torch.stack([b[1] for b in res], dim=0)
        scene_graph={
                        "input_ids":torch.stack([b[2]["input_ids"][0] for b in res], dim=0),
                        "attention_mask":torch.stack([b[2]["attention_mask"][0] for b in res], dim=0)
                        }
        IE_inputs={
                    "input_ids":torch.stack([b[3]["input_ids"] for b in res], dim=0),
                    "attention_mask":torch.stack([b[3]["attention_mask"] for b in res], dim=0)
                        }

        aspects_mask=[b[4] for b in res]
        aspects_scope=[b[5] for b in res]
        adj_matrix=torch.stack([b[6] for b in res], dim=0)
        aspect_targets=[b[7] for b in res]
        
        new_batch = {'image_embeds': image_embeds, 'query_inputs': query_inputs, 'scene_graph': scene_graph, 
                     'IE_inputs': IE_inputs, 'aspects_mask': aspects_mask, 'aspects_scope': aspects_scope, 
                     'adj_matrix': adj_matrix, 'aspect_targets': aspect_targets}
        
        false_image_embeds=torch.stack([b[0] for b in false_res], dim=0)
        false_query_inputs=torch.stack([b[1] for b in false_res], dim=0)
        false_scene_graph={
                        "input_ids":torch.stack([b[2]["input_ids"][0] for b in false_res], dim=0),
                        "attention_mask":torch.stack([b[2]["attention_mask"][0] for b in false_res], dim=0)
                        }
        false_IE_inputs={
                    "input_ids":torch.stack([b[3]["input_ids"] for b in false_res], dim=0),
                    "attention_mask":torch.stack([b[3]["attention_mask"] for b in false_res], dim=0)
                        }

        false_aspects_mask=[b[4] for b in false_res]
        false_aspects_scope=[b[5] for b in false_res]
        false_adj_matrix=torch.stack([b[6] for b in false_res], dim=0)
        false_aspect_targets=[b[7] for b in false_res]
        
        false_batch = {'image_embeds': false_image_embeds, 'query_inputs': false_query_inputs, 'scene_graph': false_scene_graph, 
                     'IE_inputs': false_IE_inputs, 'aspects_mask': false_aspects_mask, 'aspects_scope': false_aspects_scope, 
                     'adj_matrix': false_adj_matrix, 'aspect_targets': false_aspect_targets}
        
        loss_cls_cl = 0
        class_stats = None
        return loss_cls_cl, n_correct, n_pred, n_label, new_batch, false_batch, class_stats
    
    def masc_cl_loss(self, samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs):
        adj_ag = None

        # Attention matrix
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i].clone()
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        # 去除邻接矩阵的对角线元素，并添加自环，最后与文本编码器的注意力矩阵进行元素相乘。
        adj_ag_modified = []
        for j in range(adj_ag.size(0)):
            temp = adj_ag[j] - torch.diag(torch.diag(adj_ag[j]))
            temp = temp + torch.eye(temp.size(0), device=adj_ag.device)
            adj_ag_modified.append(temp)
        adj_ag = torch.stack(adj_ag_modified)
        adj_ag = text_encoder_atts.transpose(1, 2) * adj_ag

        H_l = gcn_inputs
        si = nn.Sigmoid()
        relu = nn.ReLU()
        for l in range(self.layers):
            # **********GCN*********
            AH_sem = adj_ag.bmm(H_l)
            I_sem = self.semW[l](AH_sem) #SemGCN
            AH_dep = samples['adj_matrix'].bmm(H_l)
            I_dep = self.depW[l](AH_dep) #depGCN

            g = si(I_dep)
            I_dep_g = self.hyper1 * g # [B, S, 768/2]
            I_com = torch.mul((1-I_dep_g),I_sem) + torch.mul(I_dep_g,I_dep) # adaptive fusion
            H_out = relu(self.fc3(I_com))
            H_out = nn.LayerNorm(H_out.size(-1), device=H_out.device)(H_out)
            
            if l == 0:
                H_l_original = self.fc4(H_l)
            g_l = si(H_l_original)
            H_l = torch.mul(g_l, H_out) + torch.mul((1 - g_l),H_l_original)

        # span-masked graphCL
        h1 = self.projection(H_l)
        h2 = self.projection(I_sem)  #[B,s,D/2]
        if no_its_and_itm:
            loss_cl = 0
        else:
            l1 = self.scope_semi_loss_list(h1, h2, samples['aspects_scope'], samples['aspects_mask'])
            l2 = self.scope_semi_loss_list(h2, h1, samples['aspects_scope'], samples['aspects_mask'])  # B, Seq
            loss = (l1 + l2) * 0.5
            loss_avg = loss.mean(dim=1, keepdim=True)
            loss_cl = loss_avg.mean()

        loss_target = 0
        classes = [0, 1, 2]
        class_stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in classes}
        n_correct = 0
        n_pred = 0
        n_label = 0

        for i in range(len(samples['aspects_mask'])):
            asp_wn_ori = samples['aspects_mask'][i].sum(dim=-1).unsqueeze(-1).to(h1.device) # [N,1]
            asp_wn_ori = torch.clamp(asp_wn_ori, min=1.0)
            n_mask_ori = samples['aspects_mask'][i].unsqueeze(-1).repeat(1, 1, self.hidden_size // 2).to(h1.device)  # [N,S,D/2]
            
            h1_expanded = h1[i].unsqueeze(0)  # [1, S, D/2]
            h2_expanded = h2[i].unsqueeze(0)  # [1, S, D/2]
            masked_h1 = h1_expanded * n_mask_ori  # [N, S, D/2]
            masked_h2 = h2_expanded * n_mask_ori  # [N, S, D/2]
            summed_h1 = masked_h1.sum(dim=1)  # [N, D/2]   
            summed_h2 = masked_h2.sum(dim=1)  # [N, D/2]
            outputs1 = summed_h1 / asp_wn_ori  # [N, D/2]
            outputs2 = summed_h2 / asp_wn_ori  # [N, D/2]
            outputs1 = nn.functional.normalize(outputs1,  p=2, dim=-1)  # L2归一化 
            outputs2 = nn.functional.normalize(outputs2,  p=2, dim=-1)
            pooled_output_normalized = nn.functional.normalize(pooled_output[i],  p=2, dim=-1)

            # 合并三个输出
            final_outputs = torch.cat((outputs1, outputs2, pooled_output_normalized.repeat(outputs2.size(0), 1)), dim=-1)
            logits = self.classifier(final_outputs)  # [N, 3]
            loss_target += self.criterion(logits, samples['aspect_targets'][i].to(h1.device))
            
            labels = samples['aspect_targets'][i].to(h1.device)
            predictions = torch.argmax(logits, dim=-1)  # [N]

            for cls in classes:
                tp_mask = (predictions == cls) & (labels == cls)
                fp_mask = (predictions == cls) & (labels != cls)
                fn_mask = (predictions != cls) & (labels == cls)
                class_stats[cls]['tp'] += torch.sum(tp_mask).item() 
                class_stats[cls]['fp'] += torch.sum(fp_mask).item() 
                class_stats[cls]['fn'] += torch.sum(fn_mask).item()

            n_correct += torch.sum(predictions == labels).item()
            n_pred += samples['aspects_mask'][i].size(0)
            n_label += samples['aspects_mask'][i].size(0)
        
        if no_its_and_itm:
            loss_cls_cl = 0
        else:
            loss_classify = loss_target.mean()
            loss_cls_cl = loss_classify +  self.hyper3 * loss_cl
        
        return loss_cls_cl, n_correct, n_pred, n_label, class_stats
    
    def forward(self, samples, no_its_and_itm=False):
        PQformer_outputs = self.pdq(samples, no_its_and_itm)  # torch.Size([6, 32, 768])
        query_outputs = self.FSUIE_proj(PQformer_outputs.FSUIE_inputs)  # torch.Size([6, 32, 768])
        query_outputs = self.dropout_layer(query_outputs)
        text_attn = torch.ones(query_outputs.size()[:-1], dtype=torch.long).to(query_outputs.device)
        text_input_ids = samples['IE_inputs']['input_ids']
        text_att_mask = samples['IE_inputs']['attention_mask']
        text_encoder_atts = torch.cat([text_attn, text_att_mask], dim=1)  # torch.Size([6, 512])
        text_inputs_embeds = self.text_encoder.encoder.embeddings(input_ids=text_input_ids)
        text_inputs_embeds = torch.cat([query_outputs, text_inputs_embeds], dim=1)  # torch.Size([6, 512, 768])

        sequence_output,pooled_output,_ = self.text_encoder(inputs_embeds= text_inputs_embeds, 
                                          attention_mask=text_encoder_atts)
        
        gcn_inputs = sequence_output
        pooled_output = self.pooled_drop(pooled_output)
        text_encoder_atts = text_encoder_atts.unsqueeze(-2)
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, text_encoder_atts)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]

        if self.task == "MATE":
            loss_cls_cl, n_correct, n_pred, n_label, class_stats = self.mate_cl_loss(samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs)
            new_batch = None
            false_batch = None
        elif self.task == "MASC":
            loss_cls_cl, n_correct, n_pred, n_label, class_stats = self.masc_cl_loss(samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs)
            new_batch = None
            false_batch = None
        elif self.task == "MABSA":
            loss_cls_cl, n_correct, n_pred, n_label, new_batch, false_batch, class_stats = self.mabsa_mate(samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs)

        total_loss = (self.itc_weight * PQformer_outputs.loss_itc
                      + self.itm_weight * PQformer_outputs.loss_itm
                      + self.lm_weight * PQformer_outputs.loss_lm
                      + self.cl_weight * loss_cls_cl
                      )
        
        return DASCO_Output(
            total_loss=total_loss,
            loss_cl=loss_cls_cl,
            loss_itm=PQformer_outputs.loss_itm,
            loss_itc=PQformer_outputs.loss_itc,
            loss_lm=PQformer_outputs.loss_lm,
            n_correct = n_correct,
            n_pred = n_pred,
            n_label = n_label,
            class_stats = class_stats,
            new_batch = new_batch,
            false_batch = false_batch
        )


def from_pretrained(path, args):
    pretrain_config = {
        "text_model": {"model_path": "./Text_encoder/model_best",
                       "tokenizer_path": "./Text_encoder/model_best",
                       "hidden_size": 768
                       },
        "pdq": {
            "hidden_size": 768
        },
        "loss_weights": {"itc": 1.0, "itm": 1.0, "lm":1.0, "cl": 1.0},
        "rand_seed": 0,
        "lr": 5e-5
    }
    model = DASCO(args, pretrain_config)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    print(f"loading model finished from {path}")
    return model
