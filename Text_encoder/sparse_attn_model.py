import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, PretrainedConfig
from typing import Optional
from fsa import FSA_layer

class Text_encoder(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super(Text_encoder, self).__init__(config)
        self.encoder = BertModel(config)
        self.config = config
        hidden_size = self.config.hidden_size
        adapt_span_params={'adapt_span_enabled':True,'adapt_span_loss':0.0,'adapt_span_ramp':32,'adapt_span_init':0.0,'adapt_span_cache':False}
        self.sparse_attn_layer=FSA_layer(hidden_size=hidden_size, nb_heads=8, attn_span=30, dropout=0.1, inner_hidden_size=hidden_size, adapt_span_params=adapt_span_params)
        self.sigmoid = nn.Sigmoid()
        self.BCE_loss=nn.BCEWithLogitsLoss(reduction="sum")
        self.post_init()

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict

        )
        sequence_output = outputs.last_hidden_state
        attentions=outputs.attentions
        pooled_output = outputs.pooler_output
        if output_attentions:
            sequence_output,fuzzy_span_attentions = self.sparse_attn_layer(sequence_output,output_attentions=output_attentions)
        else:
            sequence_output = self.sparse_attn_layer(sequence_output,output_attentions=output_attentions)
            fuzzy_span_attentions=None
        
        return sequence_output,pooled_output,attentions