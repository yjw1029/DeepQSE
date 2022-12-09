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
            os.path.join(self.args.bert_path, args.config_name)
            if self.args.bert_name is None
            else self.args.bert_name,
            output_hidden_states=True,
            return_dict=False,
            skip_word_embedding=True,
        )
        # manually change layer from config
        self.config.vocab_size = 10
        self.config.num_hidden_layers = args.bert_output_layer
        if args.model_type == "minitulr":
            self.unicoder = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path)
                if self.args.bert_name is None
                else self.args.bert_name,
                prefix="aggregator.",
                strip_prefix=True,
                config=self.config,
                skip_word_embedding=True,
            )
        else:
            self.unicoder = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path)
                if self.args.bert_name is None
                else self.args.bert_name,
                config=self.config,
                skip_word_embedding=True,
            )
        self.drop_layer = nn.Dropout(p=args.drop_rate)

    def forward(self, sent_embs, sent_mask, position_ids):
        if self.args.model_type == "tnlr":
            sent_vecs = self.unicoder(
                inputs_embeds=sent_embs,
                attention_mask=sent_mask,
                position_ids=position_ids,
            )[3][self.args.bert_output_layer]
        else:
            sent_vecs = self.unicoder(
                inputs_embeds=sent_embs,
                attention_mask=sent_mask,
                position_ids=position_ids,
            )[2][self.args.bert_output_layer]
        return sent_vecs


class TopKModel(nn.Module):
    def __init__(self, args):
        super(TopKModel, self).__init__()
        self.args = args

        self.sents_encoder = TextEncoder(args)
        self.sents_agg = SentAgg(args)

        self.pred_linear = nn.Sequential(
            nn.Linear(
                self.sents_encoder.config.hidden_size,
                self.sents_encoder.config.hidden_size // 2,
            ),
            nn.Tanh(),
            nn.Linear(self.sents_encoder.config.hidden_size // 2, 1),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, compute_loss=True):

        batch_size = inputs.get("queries").get("input_ids").shape[0]
        bodies = inputs.get("bodies")

        body_sents_embs = self.sents_encoder(bodies)
        body_embs = torch.index_select(body_sents_embs, 0, inputs.get("body_indexes"))
        body_embs = body_embs.view(batch_size, -1, body_sents_embs.shape[-1])

        query_embs = self.sents_encoder(inputs.get("queries")).unsqueeze(1)

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
