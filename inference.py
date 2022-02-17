# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan
# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import torch
import utils
import timeit
import argparse

from typing import Any
from constants import *
from model import MRCBertModel
from torch.nn import functional as F
from transformers import BertTokenizer
from unicodedata import normalize as unormalize


def preprocessing(input: str=None):
    input = unormalize('NFKC', input)
    input = input.lower().strip()

    return input


def infer(input: str=None, tokenizer: Any=None, model: Any=None, inference_beta: float=0.8):
    model.eval()

    asp_predict = []
    opi_predict = []
    asp_opi_predict = []
    asp_pol_predict = []

    forward_pair_list = []
    forward_pair_prob = []
    forward_pair_ind_list = []

    final_asp_list = []
    final_opi_list = []
    final_asp_ind_list = []
    final_opi_ind_list = []

    #TODO: Pre-processing
    input = preprocessing(input)
    word_list = input.split()

    #TODO: Forward
    f_asp_query = f"[CLS] What aspects ? [SEP] {input}".split()
    forward_asp_query_seg = [0]*(len(f_asp_query) - len(word_list)) + [1]*len(word_list)
    forward_asp_query = tokenizer.convert_tokens_to_ids(
        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_asp_query]
    )
    forward_asp_query_mask = [1 for i in range(len(f_asp_query))]

    forward_asp_query = torch.tensor(forward_asp_query).unsqueeze(0).long().cuda()
    forward_asp_query_seg = torch.tensor(forward_asp_query_seg).unsqueeze(0).long().cuda()
    forward_asp_query_mask = torch.tensor(forward_asp_query_mask).unsqueeze(0).float().cuda()

    f_asp_start_scores, f_asp_end_scores = model(forward_asp_query,
                                                 forward_asp_query_mask,
                                                 forward_asp_query_seg, 0)

    f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
    f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
    f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
    f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

    f_asp_start_prob_temp = []
    f_asp_end_prob_temp = []
    f_asp_start_index_temp = []
    f_asp_end_index_temp = []

    for i in range(f_asp_start_ind.size(0)):
        if f_asp_start_ind[i].item() == 1:
            f_asp_start_index_temp.append(i)
            f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
        if f_asp_end_ind[i].item() == 1:
            f_asp_end_index_temp.append(i)
            f_asp_end_prob_temp.append(f_asp_end_prob[i].item())

    f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
        f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)

    for i in range(len(f_asp_start_index)):
        asp = [forward_asp_query[0][j].item() for j in
               range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
        asp_ind = [f_asp_start_index[i] - 5, f_asp_end_index[i] - 5]
        temp_prob = f_asp_prob[i]

        final_asp_list.append(asp)
        final_asp_ind_list.append(asp_ind)

    # sentiment
    for idx in range(len(final_asp_list)):
        sentiment_query = tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
             '[CLS] What sentiment given the aspect'.split(' ')])
        sentiment_query += final_asp_list[idx]
        sentiment_query.append(tokenizer.convert_tokens_to_ids('?'))
        sentiment_query.append(tokenizer.convert_tokens_to_ids('[SEP]'))

        sentiment_query_seg = [0] * len(sentiment_query)
        sentiment_query = torch.tensor(sentiment_query).long().cuda()
        sentiment_query = torch.cat([sentiment_query, forward_asp_query[0][5:]], -1).unsqueeze(0)
        sentiment_query_seg += [1] * forward_asp_query[0][5:].size(0)
        sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
        sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

        sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
        sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

        asp_f = []
        asp_f.append(final_asp_ind_list[idx][0])
        asp_f.append(final_asp_ind_list[idx][1])
        if asp_f + [sentiment_predicted] not in asp_pol_predict:
            asp_pol_predict.append(asp_f + [sentiment_predicted])

    real_output = []
    for asp_pol in asp_pol_predict:
        real_output.append([convert_ids_to_text(asp_pol[: 2], input), ID2SENTIMENT[asp_pol[-1]]])

    return real_output


def convert_ids_to_text(ids, text):
    return ' '.join(text.split()[ids[0]: ids[1] + 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to the test file.')
    parser.add_argument('--input', type=str, default=None,
                        help='The text input.')
    parser.add_argument('--model_type', type=str, default="bert-base-uncased")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.8)
    parser.add_argument('--model_file_path', type=str, default='./models/best_model.pt',
                        help='Path to the pretrained model.')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    model = MRCBertModel(args)
    print(f'Loading model path: `{args.model_file_path}`.')
    checkpoint = torch.load(args.model_file_path)
    model.load_state_dict(checkpoint['net'])
    model = model.cuda()

    args.input = "Owner is pleasant and entertaining ."

    if not args.test_path and not args.input:
        raise ValueError(f"Must be a given text input or file path input.")

    if args.input:
        start_time = timeit.default_timer()
        output = infer(args.input, tokenizer, model, args.inference_beta)
        end_time = timeit.default_timer()
        print(f"Output: {output}")
        print(f"Inference time: {(end_time - start_time) * 1000} ms.")

    if args.test_path:
        with open(args.test_path, 'r', encoding='utf-8') as f:
            text_lines = [line.split('####')[0].strip() for line in f.readlines()]

        inference_times = []
        for text in text_lines:
            start_time = timeit.default_timer()
            output = infer(args.input, tokenizer, model, args.inference_beta)
            end_time = timeit.default_timer()
            inference_times.append((end_time - start_time) * 1000)

        print(f"Inference time average on Test: {sum(inference_times) / len(inference_times)} ms.")
