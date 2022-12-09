import os
import logging
import torch
from torch import nn
import torch.nn.functional as F

from deepqse.model.toptf_load import SentAgg
from deepqse.utils import MODEL_CLASSES


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args

        # init plm for text encoder
        assert "cross" in args.model_type, "Must be cross PLM for QueryKVModel"

        config_class, model_class, _ = MODEL_CLASSES[args.model_type]

        self.config = config_class.from_pretrained(
            os.path.join(self.args.bert_path, args.config_name)
            if self.args.bert_name is None
            else self.args.bert_name,
            output_hidden_states=True,
            return_dict=False,
        )

        self.config.num_hidden_layers = args.bert_output_layer

        if "minitulr" in args.model_type:
            self.query_unicoder, loading_info = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path)
                if self.args.bert_name is None
                else self.args.bert_name,
                prefix="encoder.",
                strip_prefix=True,
                config=self.config,
                output_loading_info=True,
            )
            self.sent_unicoder, loading_info = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path)
                if self.args.bert_name is None
                else self.args.bert_name,
                prefix="encoder.",
                strip_prefix=True,
                config=self.config,
                output_loading_info=True,
            )
        else:
            self.query_unicoder, loading_info = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path)
                if self.args.bert_name is None
                else self.args.bert_name,
                config=self.config,
                output_loading_info=True,
            )
            self.sent_unicoder, loading_info = model_class.from_pretrained(
                os.path.join(self.args.bert_path, args.model_name_or_path)
                if self.args.bert_name is None
                else self.args.bert_name,
                config=self.config,
                output_loading_info=True,
            )

        self.log_loading_info(loading_info)

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

    def layer_forward(
        self,
        query_layer_module,
        sent_layer_module,
        query_hidden_states,
        query_extended_attention_mask,
        sents_hidden_states,
        sents_extended_attention_mask,
        query_indexes,
    ):

        query_layer_output = query_layer_module(
            hidden_states=query_hidden_states,
            attention_mask=query_extended_attention_mask,
            output_qkv=True,
        )

        query_hidden_states, (query_key, query_value) = query_layer_output

        sents_query_key = torch.index_select(query_key, 0, query_indexes)
        sents_query_value = torch.index_select(query_value, 0, query_indexes)
        query_in_sents_extended_attention_mask = torch.index_select(
            query_extended_attention_mask, 0, query_indexes
        )

        sents_extended_attention_mask = torch.cat(
            [query_in_sents_extended_attention_mask, sents_extended_attention_mask],
            dim=3,
        )

        sents_layer_output = sent_layer_module(
            hidden_states=sents_hidden_states,
            attention_mask=sents_extended_attention_mask,
            past_key_value=(sents_query_key, sents_query_value),
        )

        return query_layer_output, sents_layer_output

    def forward(
        self,
        sents_input_ids,
        sents_attention_mask,
        queries_input_ids,
        queries_attention_mask,
        queries_indexes,
    ):
        # sents = inputs.get("bodies")
        # sents_input_ids = sents.get("input_ids")
        # sents_attention_mask = sents.get("attention_mask")
        sents_extended_attention_mask = self.sent_unicoder.get_extended_attention_mask(
            sents_attention_mask, sents_input_ids.size(), sents_input_ids.device
        )

        # queries = inputs.get("queries")
        # queries_input_ids = queries.get("input_ids")
        # queries_attention_mask = queries.get("attention_mask")
        queries_extended_attention_mask = (
            self.query_unicoder.get_extended_attention_mask(
                queries_attention_mask,
                queries_input_ids.size(),
                queries_input_ids.device,
            )
        )

        # embddings
        sents_embedding_output = self.sent_unicoder.embeddings(
            input_ids=sents_input_ids
        )
        queries_embedding_output = self.query_unicoder.embeddings(
            input_ids=queries_input_ids
        )

        queries_hidden_states, sents_hidden_state = (
            queries_embedding_output,
            sents_embedding_output,
        )
        # queries_indexes = inputs.get("query_indexes")
        for i, (query_layer_module, sent_layer_module) in enumerate(
            zip(self.query_unicoder.encoder.layer, self.sent_unicoder.encoder.layer)
        ):
            queries_layer_output, sents_layer_output = self.layer_forward(
                query_layer_module,
                sent_layer_module,
                queries_hidden_states,
                queries_extended_attention_mask,
                sents_hidden_state,
                sents_extended_attention_mask,
                queries_indexes,
            )

            queries_hidden_states = queries_layer_output[0]
            sents_hidden_state = sents_layer_output[0]

        queries_embs = queries_hidden_states[:, 0, :]
        sents_embs = sents_hidden_state[:, 0, :]

        return queries_embs, sents_embs


class DiffQueryKVModel(nn.Module):
    def __init__(self, args):
        super(DiffQueryKVModel, self).__init__()
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
            sents_input_ids = inputs.get("bodies").get("input_ids"),
            sents_attention_mask = inputs.get("bodies").get("attention_mask"),
            queries_input_ids = inputs.get("queries").get("input_ids"),
            queries_attention_mask = inputs.get("queries").get("attention_mask"),
            queries_indexes = inputs.get("query_indexes"))

        # sents_attention_mask = sents.get("attention_mask"))

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
