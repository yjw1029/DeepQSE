import os
import torch
from torch import nn

from .toptf_load import TextEncoder, SentAgg
from ..utils import MODEL_CLASSES


class SentAgg(nn.Module):
    def __init__(self, args):
        super(SentAgg, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.config = config_class.from_pretrained(
            os.path.join(self.args.bert_path, args.config_name),
            output_hidden_states=True,
            return_dict=False,
        )
        # manually change layer from config
        self.config.vocab_size = 10
        self.config.num_hidden_layers = args.bert_output_layer
        if args.model_type == "minitulr":
            self.unicoder = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path),
                prefix="aggregator.",
                strip_prefix=True,
                config=self.config,
            )
        else:
            self.unicoder = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path),
                config=self.config,
            )
        self.drop_layer = nn.Dropout(p=args.drop_rate)

    def forward(self, sent_embs, position_ids):
        if self.args.model_type == "tnlr":
            sent_vecs = self.unicoder(
                inputs_embeds=sent_embs, position_ids=position_ids
            )[3][self.args.bert_output_layer]
        else:
            sent_vecs = self.unicoder(
                inputs_embeds=sent_embs, position_ids=position_ids
            )[2][self.args.bert_output_layer]
        return sent_vecs


class TwoStepModel(nn.Module):
    def __init__(self, args):
        super(TwoStepModel, self).__init__()
        self.args = args

        self.query_encoder = TextEncoder(args)
        self.sents_encoder = self.query_encoder

        self.sents_encoder2 = TextEncoder(args)
        self.sents_agg = SentAgg(args)

        self.pred_linear = nn.Sequential(
            nn.Linear(
                self.sents_encoder2.config.hidden_size,
                self.sents_encoder2.config.hidden_size // 2,
            ),
            nn.Tanh(),
            nn.Linear(self.sents_encoder2.config.hidden_size // 2, 1),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def step1(self, inputs, compute_loss=True):
        batch_size = inputs.get("queries").get("input_ids").shape[0]
        bodies = inputs.get("bodies")

        body_sents_embs = self.sents_encoder(bodies)
        body_embs = torch.index_select(
            body_sents_embs, 0, inputs.get("body_indexes")
        ).view(batch_size, -1, body_sents_embs.shape[-1])

        query_embs = self.query_encoder(inputs.get("queries"))

        scores = (
            torch.einsum("bsd,bd->bs", [body_embs, query_embs])
            + (1 - inputs.get("sents_mask")) * -1e4
        )

        if compute_loss:
            loss = self.loss_fn(scores, inputs.get("start_indexes"))
            return loss, scores
        else:
            return scores

    def step2(self, inputs, scores_step1, compute_loss=True):
        batch_size, max_sent_num = scores_step1.shape

        # if sentense_num is less than args.step1_topk
        topk_num = min(max_sent_num, self.args.step1_topk)

        if self.training:
            # select other top k and start_index
            scores_step1[
                torch.arange(batch_size)[:, None], inputs["start_indexes"]
            ] = -1e-4
            topk_indices = torch.topk(scores_step1, k=topk_num - 1, dim=1).indices
            topk_indices = torch.cat(
                [topk_indices, inputs["start_indexes"].unsqueeze(-1)], dim=1
            )
        else:
            topk_indices = torch.topk(scores_step1, k=topk_num, dim=1).indices

        body_indexes = inputs.get("body_indexes").view(batch_size, -1)
        topk_body_indices = body_indexes[
            torch.arange(batch_size)[:, None], topk_indices
        ].view(-1)

        bodies = inputs.get("query_bodies")
        for key in bodies:
            bodies[key] = torch.index_select(bodies[key], 0, topk_body_indices)

        body_embs = self.sents_encoder(bodies).reshape(
            batch_size, topk_num, -1
        )
        query_embs = self.sents_encoder(inputs.get("queries")).unsqueeze(1)

        sents_embs = torch.cat([query_embs, body_embs], dim=1)
        position_ids = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.float32).to(sents_embs.device),
                topk_indices + 1,
            ],
            dim=1,
        ).long()
        sents_vecs = self.sents_agg(sents_embs, position_ids)
        body_vecs = sents_vecs[:, 1:, :]

        all_scores = torch.zeros_like(scores_step1) - 1e4
        scores = self.pred_linear(body_vecs).view(batch_size, -1)
        all_scores[torch.arange(batch_size)[:, None], topk_indices] += scores
        # scores_step2 = all_scores + scores_step1

        if compute_loss:
            loss = self.loss_fn(all_scores, inputs["start_indexes"])
            return loss, all_scores
        else:
            return all_scores

    def forward(self, inputs, compute_loss=True):
        if compute_loss:
            loss1, scores1 = self.step1(inputs, compute_loss)
            loss2, scores2 = self.step2(inputs, scores1.clone().detach(), compute_loss)
            loss = loss1 + loss2
            return loss, scores2
        else:
            scores1 = self.step1(inputs, compute_loss)
            scores2 = self.step2(inputs, scores1.clone().detach(), compute_loss)
            return scores2
