import math
import logging
import torch
import torch.nn as nn
from collections import OrderedDict

from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaEncoder,
    RobertaLayer,
)


class FastSelfAttention(nn.Module):
    def __init__(self, config, **kwargs):
        super(FastSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        # batch, num_head, seq_num
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / math.sqrt(
            self.attention_head_size
        )

        if attention_mask is not None:
            query_for_score += attention_mask

        query_weight = self.softmax(query_for_score).unsqueeze(2)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        pooled_query = torch.matmul(query_weight, query_layer).transpose(2, 3)

        key_layer = self.transpose_for_scores(mixed_key_layer)

        query_key_score = torch.matmul(key_layer, pooled_query).squeeze(-1) / math.sqrt(
            self.attention_head_size
        )

        if attention_mask is not None:
            query_key_score += attention_mask
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        pooled_key = torch.matmul(query_key_weight, key_layer)
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2]
            + (self.num_attention_heads * self.attention_head_size,)
        )
        weighted_value = self.transform(weighted_value) + mixed_query_layer

        outputs = (weighted_value, None) if output_attentions else (weighted_value,)
        return outputs


class FastFormerLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = FastSelfAttention(config)


class FastFormerEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList(
            [FastFormerLayer(config) for _ in range(config.num_hidden_layers)]
        )


class FastFormerModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config
        self.encoder = FastFormerEncoder(config)

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        extended_attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.0
        return extended_attention_mask.to(device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        prefix=None,
        strip_prefix=True,
        **kwargs,
    ):
        model_dict = torch.load(pretrained_model_name_or_path)
        """add prefix and strip_prefix for partially load statedict"""
        if prefix is not None:
            new_model_dict = OrderedDict()
            for key in model_dict:
                if key.startswith(prefix):
                    if strip_prefix:
                        new_model_dict[key[len(prefix) :]] = model_dict[key]
                    else:
                        new_model_dict[key] = model_dict[key]
        else:
            new_model_dict = model_dict

        """ not load position ids if max position embeddings changed """
        if kwargs["config"].max_position_embeddings != 512:
            del new_model_dict["embeddings.position_ids"]
            del new_model_dict["embeddings.position_embeddings.weight"]

            logging.info(
                "delete embeddings.position_ids and embeddings.position_embeddings.weight from state_dict"
            )

        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            state_dict=new_model_dict,
            **kwargs,
        )

