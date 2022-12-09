import os
import logging
import torch
from torch import nn
import torch.nn.functional as F

from deepqse.model.single_speed.diff_query_kv import TextEncoder
from deepqse.model.topk import SentAgg
from deepqse.utils import MODEL_CLASSES


class DiffQueryKVTopKModel(nn.Module):
    def __init__(self, args):
        super(DiffQueryKVTopKModel, self).__init__()
        self.args = args

        self.text_encoder = TextEncoder(args)
        self.sents_agg = SentAgg(args)

        self.pred_linear = nn.Sequential(
            nn.Linear(
                self.text_encoder.config.hidden_size,
                self.text_encoder.config.hidden_size // 2,
            ),
            nn.Tanh(),
            nn.Linear(self.text_encoder.config.hidden_size // 2, 1),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.drop_layer = nn.Dropout(p=args.drop_rate)

    def forward(self, inputs, compute_loss=True):
        # one for padding query
        batch_size = inputs.get("queries").get("input_ids").shape[0] - 1
        query_embs, body_sents_embs = self.text_encoder(
            sents_input_ids=inputs.get("bodies").get("input_ids"),
            sents_attention_mask=inputs.get("bodies").get("attention_mask"),
            queries_input_ids=inputs.get("queries").get("input_ids"),
            queries_attention_mask=inputs.get("queries").get("attention_mask"),
            queries_indexes=inputs.get("query_indexes"),
        )

        body_embs = torch.index_select(body_sents_embs, 0, inputs.get("body_indexes"))
        body_embs = body_embs.view(batch_size, -1, body_sents_embs.shape[-1])

        # remove padding query
        query_embs = query_embs[1:, :].unsqueeze(1)

        sents_embs = torch.cat([query_embs, body_embs], dim=1)
        sents_mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.float32).to(sents_embs.device),
                inputs.get("sents_mask"),
            ],
            dim=1,
        )
        position_ids = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.int32).to(sents_embs.device),
                inputs.get("pos_ids"),
            ],
            dim=1,
        ).long()
        sents_vecs = self.sents_agg(sents_embs, sents_mask, position_ids)
        body_vecs = sents_vecs[:, 1:, :]

        scores = (
            self.pred_linear(body_vecs).view(batch_size, -1)
            + (1 - inputs.get("sents_mask")) * -1e4
        )
        if compute_loss:
            loss = self.loss_fn(scores, inputs["start_indexes"])
            return loss, scores
        else:
            return scores
