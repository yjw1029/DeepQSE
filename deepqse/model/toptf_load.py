import os
import logging
import torch
from torch import nn
import torch.nn.functional as F

from deepqse.model.single import TextEncoder
from deepqse.utils import MODEL_CLASSES


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
        )
        # manually change layer from config
        self.config.vocab_size = 10
        self.config.num_hidden_layers = args.bert_output_layer
        if "minitulr" in args.model_type:
            self.unicoder, loading_info = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path)
                if self.args.bert_name is None
                else self.args.bert_name,
                prefix="aggregator.",
                strip_prefix=True,
                config=self.config,
                output_loading_info=True,
                skip_word_embedding=True
            )
        else:
            self.unicoder, loading_info = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path) 
                if self.args.bert_name is None
                else self.args.bert_name,
                config=self.config,
                output_loading_info=True,
                skip_word_embedding=True
            )

        self.log_loading_info(loading_info)
        self.drop_layer = nn.Dropout(p=args.drop_rate)

    def log_loading_info(self, loading_info):
        logging.debug("missing keys:")
        for key in loading_info["missing_keys"]:
            logging.debug(key)

        logging.debug("unexpected keys:")
        for key in loading_info["unexpected_keys"]:
            logging.debug(key)

        logging.debug("mismatched keys:")
        for key in loading_info["mismatched_keys"]:
            logging.debug(key)

        logging.debug("error msgs:")
        for key in loading_info["error_msgs"]:
            logging.debug(key)

    def forward(self, sent_embs, sents_mask):
        if self.args.model_type == "tnlr":
            sent_vecs = self.unicoder(
                inputs_embeds=sent_embs, attention_mask=sents_mask
            )[3][self.args.bert_output_layer]
        else:
            sent_vecs = self.unicoder(
                inputs_embeds=sent_embs, attention_mask=sents_mask
            )[2][self.args.bert_output_layer]
        return sent_vecs


class TopTFModelLoad(nn.Module):
    def __init__(self, args):
        super(TopTFModelLoad, self).__init__()
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
        sents_vecs = self.sents_agg(sents_embs, sents_mask)
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
