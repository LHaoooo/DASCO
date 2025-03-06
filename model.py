import sys
sys.path.append('Text_encoder')
sys.path.append('PDQ')
from transformers import BertTokenizerFast
from transformers import AutoModel, AutoImageProcessor
from Text_encoder.sparse_attn_model import Text_encoder_with_epe
from PDQ.PDQ import PDQ
from transformers.utils import ModelOutput
from typing import Optional, Tuple
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


def get_pred_span(start_ids, end_ids):
    start_list = torch.nonzero(start_ids)
    end_list = torch.nonzero(end_ids)
    start_list = [x[0] - 32 for x in start_list]
    end_list = [x[0] - 32 + 1 for x in end_list]
    text_span = []
    if len(start_list) == 0 or len(end_list) == 0:
        return []
    i = 0
    j = 0
    while i < len(start_list) and j < len(end_list):
        if start_list[i] < end_list[j]:
            text_span.append([start_list[i], start_list[i]])
            i += 1
        else:
            j += 1
    return text_span

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
        self.text_encoder = Text_encoder_with_epe.from_pretrained(MFSUIE_config['text_model']["model_path"])
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

        self.attention_heads = 8
        self.mem_dim = self.hidden_size // 2
        self.attn = MultiHeadAttention(self.attention_heads, self.hidden_size)
        self.layernorm = LayerNorm(self.hidden_size)
        self.pooled_drop = nn.Dropout(0.3)
        
        # 双路GCN
        self.depW = nn.ModuleList() # DepGCN依存句法GCN
        for layer in range(self.layers):
            input_dim = text_hidden_size if layer == 0 else self.mem_dim
            self.depW.append(nn.Linear(input_dim, self.mem_dim))
            # input_dim = self.hidden_size
            # self.depW.append(nn.Linear(input_dim, input_dim))
            
        self.semW = nn.ModuleList()  # SemGCN语义GCN
        for j in range(self.layers):
            input_dim = text_hidden_size if j == 0 else self.mem_dim
            self.semW.append(nn.Linear(input_dim, self.mem_dim))
            # input_dim = self.hidden_size
            # self.semW.append(nn.Linear(input_dim, input_dim))
        
        self.fc1 = torch.nn.Linear(self.hidden_size//2, 32)
        self.fc2 = torch.nn.Linear(32, self.hidden_size//2)
        self.fc3 = nn.Linear(self.hidden_size//2, self.hidden_size//2)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.sigmoid = nn.Sigmoid()
        if self.task == 'MATE':
            self.classifier = nn.Linear(self.hidden_size*2, 2)
        elif self.task == 'MASC':
            self.classifier = nn.Linear(self.hidden_size*2, 3)
        self.criterion = nn.CrossEntropyLoss()

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
                adj_ag = attn_adj_list[i]
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        # 去除邻接矩阵的对角线元素，并添加自环，最后与文本编码器的注意力矩阵进行元素相乘。
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
            I_dep_g = self.hyper1 * g # [16, 100, 768/2]
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
            n_correct += torch.sum((predictions == labels) & (labels == 1)).item()
            n_pred += torch.sum(predictions == 1).item()
            n_label += torch.sum(labels == 1).item()
        
        if no_its_and_itm:
            loss_cls_cl = 0
        else:
            loss_classify = loss_target.mean()
            loss_cls_cl = loss_classify +  self.hyper3 * loss_cl

        return loss_cls_cl, n_correct, n_pred, n_label
    
    def masc_cl_loss(self, samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs):
        adj_ag = None

        # Attention matrix
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        # 去除邻接矩阵的对角线元素，并添加自环，最后与文本编码器的注意力矩阵进行元素相乘。
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
            I_dep_g = self.hyper1 * g # [16, 100, 768/2]
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
        if no_its_and_itm:
            l1 = 0
            l2 = 0
            loss_avg = 0
            loss_cl = 0
        else:
            l1 = self.scope_semi_loss_list(h1, h2, samples['aspects_scope'], samples['aspects_mask'])
            l2 = self.scope_semi_loss_list(h2, h1, samples['aspects_scope'], samples['aspects_mask'])  # B, Seq
            loss = (l1 + l2) * 0.5
            loss_avg = loss.mean(dim=1, keepdim=True)
            loss_cl = loss_avg.mean()

        loss_target = 0
        n_correct = 0
        n_pred = 0
        n_label = 0
        for i in range(len(samples['aspects_mask'])):
            asp_wn_ori = samples['aspects_mask'][i].sum(dim=-1).unsqueeze(-1).to(h1.device) # [N,1]
            asp_wn_ori = torch.clamp(asp_wn_ori, min=1.0)
            n_mask_ori = samples['aspects_mask'][i].unsqueeze(-1).repeat(1, 1, self.hidden_size // 2).to(h1.device)  # [N,S,D/2]
            
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
            loss_target += self.criterion(logits, samples['aspect_targets'][i].to(h1.device))
            
            labels = samples['aspect_targets'][i].to(h1.device)
            predictions = torch.argmax(logits, dim=-1)
            # POS
            n_correct += torch.sum((predictions == labels) & (labels == 2)).item()
            n_pred += torch.sum(predictions == 2).item()
            n_label += torch.sum(labels == 2).item()
            # NEU
            n_correct += torch.sum((predictions == labels) & (labels == 1)).item()
            n_pred += torch.sum(predictions == 1).item()
            n_label += torch.sum(labels == 1).item()
            # NEG
            n_correct += torch.sum((predictions == labels) & (labels == 0)).item()
            n_pred += torch.sum(predictions == 0).item()
            n_label += torch.sum(labels == 0).item()
        
        if no_its_and_itm:
            loss_cls_cl = 0
        else:
            loss_classify = loss_target.mean()
            loss_cls_cl = loss_classify +  self.hyper3 * loss_cl

        return loss_cls_cl, n_correct, n_pred, n_label
    
    def forward(self, samples, no_its_and_itm=False):
        PQformer_outputs = self.pdq(samples, no_its_and_itm)  # torch.Size([6, 32, 768])
        query_outputs = self.FSUIE_proj(PQformer_outputs.FSUIE_inputs)  # torch.Size([6, 32, 768])
        query_outputs = self.dropout_layer(query_outputs)
        text_attn = torch.ones(query_outputs.size()[:-1], dtype=torch.long).to(query_outputs.device)
        text_input_ids = samples['IE_inputs']['input_ids']
        text_att_mask = samples['IE_inputs']['attention_mask']
        start_ids = samples["start_ids"]  # torch.Size([6, 512])
        end_ids = samples["end_ids"]  # torch.Size([6, 512])
        prompt_mask = samples["prompt_mask"]  # torch.Size([6, 512, 512])
        text_encoder_atts = torch.cat([text_attn, text_att_mask], dim=1)  # torch.Size([6, 512])
        text_inputs_embeds = self.text_encoder.encoder.embeddings(input_ids=text_input_ids)
        text_inputs_embeds = torch.cat([query_outputs, text_inputs_embeds], dim=1)  # torch.Size([6, 512, 768])

        sequence_output,pooled_output,_ = self.text_encoder(inputs_embeds= text_inputs_embeds, 
                                          attention_mask=text_encoder_atts,
                                          start_positions=start_ids,
                                          end_positions=end_ids,
                                          prompt_mask=prompt_mask)
        
        gcn_inputs = sequence_output
        pooled_output = self.pooled_drop(pooled_output)
        text_encoder_atts = text_encoder_atts.unsqueeze(-2)
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, text_encoder_atts)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]

        if self.task == "MATE":
            loss_cls_cl, n_correct, n_pred, n_label = self.mate_cl_loss(samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs)
        elif self.task == "MASC":
            loss_cls_cl, n_correct, n_pred, n_label = self.masc_cl_loss(samples, no_its_and_itm, attn_adj_list, text_encoder_atts, pooled_output, gcn_inputs)
        elif self.task == "JMASA":
            loss_cls_cl = 0

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
            n_label = n_label
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
