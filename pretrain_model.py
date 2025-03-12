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
    accuracy_aspect: Optional[torch.FloatTensor] = None
    accuracy_match: Optional[torch.FloatTensor] = None

class DASCO_base(nn.Module):
    def __init__(self, args, MFSUIE_config):
        super().__init__()
        self.pdq = PDQ()
        self.text_encoder = Text_encoder.from_pretrained(MFSUIE_config['text_model']["model_path"])
        
        Qformer_hidden_size = MFSUIE_config["pdq"]["hidden_size"]
        text_hidden_size = MFSUIE_config["text_model"]["hidden_size"]
        self.hidden_size = text_hidden_size
        self.FSUIE_proj = nn.Linear(Qformer_hidden_size, text_hidden_size)
        self.itc_weight = MFSUIE_config["loss_weights"]["itc"]
        self.itm_weight = MFSUIE_config["loss_weights"]["itm"]
        self.lm_weight = MFSUIE_config["loss_weights"]["lm"]
        self.cls_weight = MFSUIE_config["loss_weights"]["cls"]
        self.dropout_layer = nn.Dropout()

        self.classifier1 = nn.Linear(text_hidden_size, 2)
        self.classifier2 = nn.Linear(text_hidden_size, 2)
    
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
        
        
        prediction_aspect = self.classifier1(sequence_output)
        prediction_match = self.classifier2(sequence_output)

        p_aspect = torch.argmax(prediction_aspect, dim=-1)
        n_correct_aspect = (p_aspect == samples['aspect_exist_targets']).sum().item()
        n_label_aspect = samples['aspect_exist_targets'].size(0)
        accuracy_aspect = n_correct_aspect / n_label_aspect

        p_match = torch.argmax(prediction_match, dim=-1)
        n_correct_match = (p_match == samples['match_target']).sum().item()
        n_label_match = samples['match_target'].size(0)
        accuracy_match = n_correct_match / n_label_match

        loss_cls_aspect = F.cross_entropy(prediction_aspect, samples['aspect_exist_targets'])
        loss_cls_match = F.cross_entropy(prediction_match, samples['match_target'])
        
        loss_cls = loss_cls_aspect + loss_cls_match
        total_loss = (self.itc_weight * PQformer_outputs.loss_itc
                      + self.itm_weight * PQformer_outputs.loss_itm
                      + self.lm_weight * PQformer_outputs.loss_lm
                      + self.cls_weight * loss_cls
                      )
        
        return DASCO_Output(
            total_loss=total_loss,
            loss_cl=loss_cls,
            loss_itm=PQformer_outputs.loss_itm,
            loss_itc=PQformer_outputs.loss_itc,
            loss_lm=PQformer_outputs.loss_lm,
            accuracy_aspect=accuracy_aspect,
            accuracy_match=accuracy_match
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
        "loss_weights": {"itc": 1.0, "itm": 1.0, "lm":1.0, "cls": 1.0},
        "rand_seed": 0,
        "lr": 5e-5
    }
    model = DASCO_base(args, pretrain_config)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    print(f"loading model finished from {path}")
    return model