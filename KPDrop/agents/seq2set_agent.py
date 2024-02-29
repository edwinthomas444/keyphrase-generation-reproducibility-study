import torch as T
import torch.nn as nn
from torch.optim import *
from utils.evaluation_utils import evaluate
import copy
import math
import nltk
import torch.nn.functional as F
from nltk.stem import PorterStemmer
import numpy as np


class seq2set_agent:
    def __init__(self, model, config, device):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = eval(config["optimizer"])
        if config["custom_betas"]:
            self.optimizer = optimizer(self.parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"],
                                       betas=(0.9, 0.998), eps=1e-09)
        else:
            # weight decay=0.0 in seq2set config
            # starting lr is 1e-03 in default config
            self.optimizer = optimizer(self.parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"])

        self.config = config
        self.eos_id = config["vocab2idx"]["<eos>"]
        self.null_id = config["NULL_id"]
        self.key = "none"
        # self.label_smoothing = self.config["label_smoothing"]
        self.device = device
        self.DataParallel = config["DataParallel"]
        self.optimizer.zero_grad()
        self.vocab2idx = config["vocab2idx"]
        self.idx2vocab = {id: token for token, id in self.vocab2idx.items()}
        self.vocab_len = len(config["vocab2idx"])
        self.eps = 1e-9
        self.epoch_level_scheduler = True
        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                mode='max',
                                                                factor=0.5,
                                                                patience=config["scheduler_patience"])

    def loss_fn(self, logits, labels, output_mask, penalty_item=None):
        # print('\n labels:' , labels.detach().cpu().tolist())

        # logits is after softmax
        vocab_len = logits.size(-1)
        N = logits.size(0)
        S1 = self.config["max_kp_num"]
        S2 = self.config["max_decoder_len"]
        #assert labels.size() == (N, S1, S2)
        #assert logits.size() == (N, S1, S2, vocab_len)
        #assert output_mask.size() == (N, S1, S2)
        #assert (logits >= 0.0).all()

        """
        true_dist = F.one_hot(labels, num_classes=vocab_len)
        #assert true_dist.size() == (N, S1, S2, vocab_len)
        #assert (true_dist >= 0).all()
        """
        ground_logit = T.gather(
            logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        #assert ground_logit.size() == (N, S1, S2)

        ground_logit = T.where(ground_logit == 0.0,
                               T.empty_like(ground_logit).fill_(
                                   self.eps).float().to(logits.device),
                               ground_logit)
        

        cross_entropy = -T.log(ground_logit)
        #assert cross_entropy.size() == (N, S1, S2)
        #assert output_mask.size() == (N, S1, S2)
        masked_cross_entropy = cross_entropy * output_mask

        if self.config["one2set"]:
            first_token_present_label = labels[:, 0:S1 // 2, 0]
            first_token_absent_label = labels[:, S1 // 2:, 0]

            prs = self.config["null_present_scale"]
            abs = self.config["null_absent_scale"]
            
            null_present_scale = T.where(first_token_present_label == self.null_id,
                                         T.empty_like(first_token_present_label).float().fill_(
                                             prs).to(labels.device),
                                         T.ones_like(first_token_present_label).float().to(labels.device))

            null_absent_scale = T.where(first_token_absent_label == self.null_id,
                                        T.empty_like(first_token_absent_label).float().fill_(
                                            abs).to(labels.device),
                                        T.ones_like(first_token_absent_label).float().to(labels.device))
            
            # non_null_units = (first_token_absent_label!= 49999).long().sum(dim=-1)
            # if non_null_units.sum(dim=-1).item()>0:
            #     print('\nnn null units batch with some non null unit: ', non_null_units)

            # print('\n ground logit: ', ground_logit)
            # print('\n null absent scale: ', null_absent_scale)
            # print('\n first token absent label: ', first_token_absent_label, self.idx2vocab[49999])
            # print('\n labels: ', labels[16, S1 // 2:, :].tolist())
            # print('\n masked cross entropy: ', masked_cross_entropy)

            null_absent_scale = null_absent_scale.unsqueeze(-1)
            null_present_scale = null_present_scale.unsqueeze(-1)
            

            #assert null_present_scale.size() == (N, S1 // 2, 1)
            #assert null_absent_scale.size() == (N, S1 // 2, 1)

            null_scale = T.cat([null_present_scale, null_absent_scale], dim=1)
            #assert null_scale.size() == (N, S1, 1)
            
            if self.config["contextualized_control_codes"]:
                penalty_item = penalty_item  # * null_scale.squeeze(-1)
            masked_cross_entropy = masked_cross_entropy * null_scale
            # output_mask = output_mask * null_scale
        elif self.config["one2one"]:
            first_token_label = labels[:, :, 0]
            null_mask = T.where(first_token_label == self.eos_id,
                                T.zeros_like(first_token_label).float().to(
                                    labels.device),
                                T.ones_like(first_token_label).float().to(labels.device))
            masked_cross_entropy = masked_cross_entropy * \
                null_mask.unsqueeze(-1)
            # output_mask = output_mask * null_mask.unsqueeze(-1)

        normalization = output_mask.sum().item()
        loss = masked_cross_entropy.sum()
        loss = loss.div(normalization)

        """
        mean_ce = T.sum(masked_cross_entropy, dim=1) / (T.sum(output_mask, dim=1) + self.eps)
        loss = T.mean(mean_ce)
        """

        if penalty_item is not None:
            loss = loss + self.config["penalty_gamma"] * penalty_item.mean()
        
        # print('\n loss: ', loss)
        return loss

    def basic_decode(self, prediction_idx, src, beam_filter=None, split_present_absent=False):
        src_token_dict = {}
        for pos, token in enumerate(src):
            if token not in src_token_dict:
                src_token_dict[token] = len(src_token_dict)

        src_token_dict_rev = {v: k for k, v in src_token_dict.items()}
        decoded_kps = []
        decoded_present_kps = []
        decoded_absent_kps = []

        max_kps = len(prediction_idx)
        assert len(prediction_idx) == 20
        for i, kp_idx in enumerate(prediction_idx):
            kp = []
            # adding
            present_kp = []
            absent_kp = []

            flag = 0
            if beam_filter is not None:
                if beam_filter[i] == 0:
                    flag = 1
            if flag == 0:
                word = None
                for id in kp_idx:
                    if id >= self.vocab_len:
                        word = src_token_dict_rev.get(
                            id - self.vocab_len, "<unk>")
                    else:
                        word = self.idx2vocab[id]
                    if word == "<eos>" or word == "<sep>" or word == ";" or word == "<null>":
                        break
                    else:
                        kp.append(word)
                if kp and word == "<eos>":
                    kp_string = " ".join(kp)
                    decoded_kps.append(kp_string)
                
                # added
                if (i+1) <= max_kps//2:
                    present_kp.extend(kp)
                else:
                    absent_kp.extend(kp)
                
                if present_kp and word == "<eos>":
                    kp_string = " ".join(kp)
                    decoded_present_kps.append(kp_string)

                if absent_kp and word == "<eos>":
                    kp_string = " ".join(kp)
                    decoded_absent_kps.append(kp_string)

        prediction = " ; ".join(decoded_kps)
        present_prediction = " ; ".join(decoded_present_kps)
        absent_prediction = " ; ".join(decoded_absent_kps)

        return prediction, decoded_kps, absent_prediction

    def decode(self, prediction_idx, src, beam_filter=None, split_present_absent=False):
        # print('prediction_idx: ', prediction_idx)

        src_token_dict = {}
        for pos, token in enumerate(src):
            if token not in src_token_dict:
                src_token_dict[token] = len(src_token_dict)

        src_token_dict_rev = {v: k for k, v in src_token_dict.items()}
        decoded_kps = []
        decoded_present_kps = []
        decoded_absent_kps = []
        # iterate through all kp_units
        max_kps = len(prediction_idx)
        # print('max_kps: ', max_kps)
        for i, kp_idx in enumerate(prediction_idx):
            kp = []
            # adding
            present_kp = []
            absent_kp = []
            flag = 0
            if beam_filter is not None:
                if beam_filter[i] == 0:
                    flag = 1
            if flag == 0:
                word = None
                max_kps = len(kp_idx)

                # iterate through words within kp
                for id in kp_idx:
                    if id >= self.vocab_len:
                        word = src_token_dict_rev.get(
                            id - self.vocab_len, "<unk>")
                    else:
                        word = self.idx2vocab[id]
                    if word == "<eos>" or word == "<sep>" or word == ";" or word == "<null>":
                        # print('kp so far: ', kp, 'current word: ', word, 'breaking')
                        break
                    else:
                        kp.append(word)

                if kp and word == "<eos>":
                    kp_string = " ".join(kp)
                    decoded_kps.append(kp_string)

                if (i+1) <= max_kps//2:
                    present_kp.extend(kp)
                else:
                    absent_kp.extend(kp)

                if present_kp and word == "<eos>":
                    kp_string = " ".join(kp)
                    decoded_present_kps.append(kp_string)

                if absent_kp and word == "<eos>":
                    kp_string = " ".join(kp)
                    decoded_absent_kps.append(kp_string)

        prediction = " ; ".join(decoded_kps)
        present_prediction = " ; ".join(decoded_present_kps)
        absent_prediction = " ; ".join(decoded_absent_kps)
        if not split_present_absent:
            return prediction, decoded_kps
        else:
            return present_prediction, decoded_present_kps, absent_prediction, decoded_absent_kps

    def run(self, batch, train=True):
        # print('\n batch size: ', batch['src_vec'].shape)

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        if not self.DataParallel:
            # print('not data parallel')
            batch["src_vec"] = batch["src_vec"].to(self.device)
            batch["trg_vec"] = batch["trg_vec"].to(self.device)
            batch["ptr_src_vec"] = batch["ptr_src_vec"].to(self.device)
            batch["src_mask"] = batch["src_mask"].to(self.device)
            batch["trg_mask"] = batch["trg_mask"].to(self.device)
            batch["labels"] = batch["labels"].to(self.device)
            if batch["first_mask"] is not None:
                batch["first_mask"] = batch["first_mask"].to(self.device)

        # print(batch["src_vec"].shape)
        # print(batch["trg_vec"].shape)
        # print(batch["ptr_src_vec"].shape)
        # print(batch['src_mask'].shape)
        # print(batch['trg_mask'].shape)
        # print(batch['labels'].shape)

        output_dict = self.model(batch)
        # print('\noutput dict prediction shape: ', output_dict['predictions'].shape)

        if self.config["generate"]:
            loss = None
        else:
            logits = output_dict["logits"]
            penalty_item = output_dict["penalty_item"]
            labels = output_dict["labels"]
            output_mask = output_dict["output_mask"]
            loss = self.loss_fn(logits=logits,
                                labels=labels,
                                output_mask=output_mask,
                                penalty_item=penalty_item)

        trgs = batch["trg"]
        display_predictions = None
        if not self.config["generate"]:
            predictions = output_dict["predictions"]
            #assert predictions.size() == labels.size()
            # if not train:
            predictions = predictions.cpu().detach().numpy().tolist()
            labels = labels.cpu().detach().numpy().tolist()
            # print('\n labels:' , labels)
            display_predictions = [self.basic_decode(prediction, src)[0] for prediction, src in
                                   zip(predictions, batch["src"])]
            predictions_temp = [self.decode(prediction, src)[0] for prediction, src in
                                zip(predictions, batch["src"])]
            trgs = [self.basic_decode(trg, src)[0].split(" ") for trg, src in
                    zip(labels, batch["src"])]

            present_predictions = [self.decode(prediction, src, split_present_absent=True)[0] for prediction, src in
                                   zip(predictions, batch["src"])]
            absent_predictions = [self.decode(prediction, src, split_present_absent=True)[2] for prediction, src in
                                  zip(predictions, batch["src"])]
            

            # print('\n One Block: \n')

            # for x,y in zip(present_predictions, absent_predictions):
            #     if x or y:
            #         print(x, '<P-----A>', y)

            predictions = predictions_temp
            # p_trgs = [self.basic_decode(trg, src, split_present_absent=True)[0].split(" ") for trg, src in
            #         zip(labels, batch["src"])]
            # a_trgs = [self.basic_decode(trg, src, split_present_absent=True)[2] for trg, src in
            #         zip(labels, batch["src"])]
            # print('\n absent target: ', a_trgs)
            # #else:
            # display_predictions = ["display_disabled"] * (batch["batch_size"])

        elif not (self.config["beam_search"] and self.config["generate"] and not self.config["top_beam"]):
            predictions = output_dict["predictions"]
            # only take the top beam's prediction
            if self.config["generate"] and self.config["top_beam"] and self.config["beam_search"]:
                N = batch["src_vec"].size(0)
                N2, S2, S3 = predictions.size()
                predictions = predictions.view(N, -1, S2, S3)
                predictions = predictions[:, 0, :, :]
            predictions = predictions.cpu().detach().numpy().tolist()
            predictions_temp = [self.decode(prediction, src)[0] for prediction, src in
                                zip(predictions, batch["src"])]

            present_predictions = [self.decode(prediction, src, split_present_absent=True)[0] for prediction, src in
                                   zip(predictions, batch["src"])]
            absent_predictions = [self.decode(prediction, src, split_present_absent=True)[2] for prediction, src in
                                  zip(predictions, batch["src"])]
            predictions = predictions_temp

        else:
            predictions = output_dict["predictions"]
            beam_filter_masks = output_dict["beam_filter_mask"]
            N2, S2, S3 = predictions.size()
            N = batch["src_vec"].size(0)
            B = N2 // N
            predictions = predictions.view(N, B, S2, S3)
            beam_filter_masks = beam_filter_masks.view(N, B, S2)
            predictions = predictions.cpu().detach().numpy().tolist()
            beam_filter_masks = beam_filter_masks.cpu().detach().numpy().tolist()

            predictions_ = []
            present_predictions_ = []
            absent_predictions_ = []

            for j in range(N):
                src = batch["src"][j]
                beam_filter_mask = beam_filter_masks[j]
                kp_predictions = []
                present_kp_predictions = []
                absent_kp_predictions = []

                if self.config["specialized_filter"]:
                    for k, beam_prediction in enumerate(predictions[j]):
                        for l, kp_idx in enumerate(beam_prediction):
                            if kp_idx[0] == self.null_id:
                                beam_filter_mask[:][l] = 0
                for k, beam_prediction in enumerate(predictions[j]):
                    _, decoded_kps = self.decode(prediction_idx=beam_prediction,
                                                 src=src,
                                                 beam_filter=beam_filter_mask[k])

                    kp_predictions = kp_predictions + decoded_kps
                    # slicing beam predictions (i.e one slice of beam prediction at a time for all kp units)
                    # can  be divided into pkp and akp
                    _, present_decoded_kps, _, absent_decoded_kps = self.decode(prediction_idx=beam_prediction,
                                                                                src=src,
                                                                                beam_filter=beam_filter_mask[k],
                                                                                split_present_absent=True)

                    present_kp_predictions = present_kp_predictions + present_decoded_kps
                    absent_kp_predictions = absent_kp_predictions + absent_decoded_kps

                predictions_.append(" ; ".join(kp_predictions))
                present_predictions_.append(" ; ".join(present_kp_predictions))
                absent_predictions_.append(" ; ".join(absent_kp_predictions))
                # print('\n\n ## Present predictions: ', present_kp_predictions, len(present_kp_predictions))

            predictions = predictions_
            present_predictions = present_predictions_
            absent_predictions = absent_predictions_

        if self.model.training:
            metrics = {"total_data": batch["batch_size"],
                       "total_present_precision": 0,
                       "total_present_recall": 0,
                       "total_absent_recall": 0,
                       "total_absent_precision": 0}
        else:
            metrics = evaluate(copy.deepcopy(batch["src"]), copy.deepcopy(batch["trg"]), copy.deepcopy(predictions),
                               beam=False, key=self.key,
                               present_predictions=copy.deepcopy(
                                   present_predictions),
                               absent_predictions=copy.deepcopy(absent_predictions))

        if loss is not None:
            metrics["loss"] = loss.item()
        else:
            metrics["loss"] = 0.0

        if display_predictions is None:
            display_predictions = predictions

        item = {"display_items": {"source": batch["src"],
                                  "target": trgs,
                                  "predictions": display_predictions,
                                  "present_predictions": present_predictions,
                                  "absent_predictions": absent_predictions},
                "loss": loss,
                "metrics": metrics,
                "stats_metrics": metrics}

        return item

    def backward(self, loss):
        loss.backward()

    def step(self):
        # for seq2set max grad norm set to 1.0
        if self.config["max_grad_norm"] is not None:
            T.nn.utils.clip_grad_norm_(
                self.parameters, self.config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()
