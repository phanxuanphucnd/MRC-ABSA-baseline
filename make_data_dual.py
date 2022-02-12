#-*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import  BertTokenizer
from dataset import DualSample, TokenizedSample, OriginalDataset


def tokenize_data(data, mode='train'):
    max_forward_asp_query_length = 0
    max_forward_opi_query_length = 0
    max_sentiment_query_length = 0

    max_aspect_num = 0
    tokenized_sample_list = []

    header_fmt = 'Tokenize data {:>5s}'
    for sample in tqdm(data, desc=f"{header_fmt.format(mode.upper())}"):
        forward_queries = []
        forward_answers = []
        sentiment_queries = []
        sentiment_answers = []

        forward_queries_seg = []
        sentiment_queries_seg = []

        if int(len(sample.sentiment_queries)) > max_aspect_num:
            max_aspect_num = int(len(sample.sentiment_queries))

        for idx in range(len(sample.forward_queries)):
            temp_query = sample.forward_queries[idx]
            temp_text = sample.text
            temp_answer = sample.forward_answers[idx]
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            temp_answer[0] = [-1] * (len(temp_query) + 2) + temp_answer[0]
            temp_answer[1] = [-1] * (len(temp_query) + 2) + temp_answer[1]

            assert len(temp_answer[0]) == len(temp_answer[1]) == len(temp_query_to) == len(temp_query_seg)

            if len(temp_query_to) > max_forward_asp_query_length:
                max_forward_asp_query_length = len(temp_query_to)

            forward_queries.append(temp_query_to)
            forward_answers.append(temp_answer)
            forward_queries_seg.append(temp_query_seg)

        for idx in range(len(sample.sentiment_queries)):
            temp_query = sample.sentiment_queries[idx]
            temp_text = sample.text
            temp_answer = sample.sentiment_answers[idx]
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)

            assert len(temp_query_to) == len(temp_query_seg)

            if len(temp_query_to) > max_sentiment_query_length:
                max_sentiment_query_length = len(temp_query_to)

            sentiment_queries.append(temp_query_to)
            sentiment_answers.append(temp_answer)
            sentiment_queries_seg.append(temp_query_seg)

        # import numpy as np
        # print(f"forward_queries: {np.shape(forward_queries)} {type(forward_queries)} | {forward_queries}")
        # print(f"forward_answers: {np.shape(forward_answers)}")
        # print(f"sentiment_queries: {np.shape(sentiment_queries)} | {sentiment_queries}")
        # print(f"sentiment_answers: {np.shape(sentiment_answers)} | {sentiment_answers}")
        # print(f"forward_queries_seg: {np.shape(forward_queries_seg)} | {forward_queries_seg}")
        # print(f"sentiment_queries_seg: {np.shape(sentiment_queries_seg)}")

        temp_sample = TokenizedSample(
            sample.original_sample, forward_queries,
            forward_answers, sentiment_queries,
            sentiment_answers, forward_queries_seg,
            sentiment_queries_seg
        )
        # print(temp_sample)
        tokenized_sample_list.append(temp_sample)

    max_attributes = {
        'mfor_asp_len': max_forward_asp_query_length,
        'max_sent_len': max_sentiment_query_length,
        'max_aspect_num': max_aspect_num
    }
    return tokenized_sample_list, max_attributes


def preprocessing(sample_list, max_len, mode='train'):
    _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _forward_asp_query = []
    _forward_asp_answer_start = []
    _forward_asp_answer_end = []
    _forward_asp_query_mask = []
    _forward_asp_query_seg = []

    _sentiment_query = []
    _sentiment_answer = []
    _sentiment_query_mask = []
    _sentiment_query_seg = []

    _aspect_num = []

    header_fmt = 'Preprocessing {:>5s}'
    for instance in tqdm(sample_list, desc=f"{header_fmt.format(mode.upper())}"):
        f_query_list = instance.forward_queries
        f_answer_list = instance.forward_answers
        f_query_seg_list = instance.forward_seg

        s_query_list = instance.sentiment_queries
        s_answer_list = instance.sentiment_answers
        s_query_seg_list = instance.sentiment_seg

        # _aspect_num: 1/2/3/...
        _aspect_num.append(int(len(s_query_list)))

        # Forward
        # Aspect
        # query
        assert len(f_query_list[0]) == len(f_answer_list[0][0]) == len(f_answer_list[0][1])
        f_asp_pad_num = max_len['mfor_asp_len'] - len(f_query_list[0])

        _forward_asp_query.append(_tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[0]]))
        _forward_asp_query[-1].extend([0] * f_asp_pad_num)

        # query_mask
        _forward_asp_query_mask.append([1 for i in range(len(f_query_list[0]))])
        _forward_asp_query_mask[-1].extend([0] * f_asp_pad_num)

        # answer
        _forward_asp_answer_start.append(f_answer_list[0][0])
        _forward_asp_answer_start[-1].extend([-1] * f_asp_pad_num)
        _forward_asp_answer_end.append(f_answer_list[0][1])
        _forward_asp_answer_end[-1].extend([-1] * f_asp_pad_num)

        # seg
        _forward_asp_query_seg.append(f_query_seg_list[0])
        _forward_asp_query_seg[-1].extend([1] * f_asp_pad_num)

        # Sentiment
        single_sentiment_query = []
        single_sentiment_query_mask = []
        single_sentiment_query_seg = []
        single_sentiment_answer = []
        for j in range(len(s_query_list)):
            sent_pad_num = max_len['max_sent_len'] - len(s_query_list[j])
            single_sentiment_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in s_query_list[j]]))
            single_sentiment_query[-1].extend([0] * sent_pad_num)

            single_sentiment_query_mask.append([1 for i in range(len(s_query_list[j]))])
            single_sentiment_query_mask[-1].extend([0] * sent_pad_num)

            # query_seg
            single_sentiment_query_seg.append(s_query_seg_list[j])
            single_sentiment_query_seg[-1].extend([1] * sent_pad_num)

            single_sentiment_answer.append(s_answer_list[j])

        _sentiment_query.append(single_sentiment_query)
        _sentiment_query[-1].extend(
            [[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_query_mask.append(single_sentiment_query_mask)
        _sentiment_query_mask[-1].extend(
            [[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_query_seg.append(single_sentiment_query_seg)
        _sentiment_query_seg[-1].extend(
            [[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_answer.append(single_sentiment_answer)
        _sentiment_answer[-1].extend([-1] * (max_len['max_aspect_num'] - _aspect_num[-1]))

    result = {
        "_forward_asp_query": _forward_asp_query,
        "_forward_asp_answer_start": _forward_asp_answer_start,
        "_forward_asp_answer_end": _forward_asp_answer_end,
        "_forward_asp_query_mask": _forward_asp_query_mask,
        "_forward_asp_query_seg": _forward_asp_query_seg,
        "_sentiment_query": _sentiment_query,
        "_sentiment_answer": _sentiment_answer,
        "_sentiment_query_mask": _sentiment_query_mask,
        "_sentiment_query_seg": _sentiment_query_seg,
        "_aspect_num": _aspect_num,
    }

    return OriginalDataset(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./data/14lap/preprocess',
                        help='Path to the processed data from `data_process.py`')
    parser.add_argument('--output_path', type=str, default='./data/14lap/preprocess',
                        help='Path to the saved data.')

    args = parser.parse_args()

    train_data_path = f"{args.data_path}/train_DUAL.pt"
    dev_data_path = f"{args.data_path}/dev_DUAL.pt"
    test_data_path = f"{args.data_path}/test_DUAL.pt"

    train_data = torch.load(train_data_path)
    dev_data = torch.load(dev_data_path)
    test_data = torch.load(test_data_path)

    train_tokenized, train_max_len = tokenize_data(train_data, mode='train')
    dev_tokenized, dev_max_len = tokenize_data(dev_data, mode='dev')
    test_tokenized, test_max_len = tokenize_data(test_data, mode='test')

    print(f"\nMax attributes")
    print(f"train_max_len : {train_max_len}")
    print(f"dev_max_len : {dev_max_len}")
    print(f"test_max_len : {test_max_len}\n")

    train_preprocess = preprocessing(train_tokenized, train_max_len, mode='train')
    dev_preprocess = preprocessing(dev_tokenized, dev_max_len, mode='dev')
    test_preprocess = preprocessing(test_tokenized, test_max_len, mode='test')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_path = f"{args.output_path}/data.pt"
    print(f"Saved data : `{output_path}`.")
    torch.save({
        'train': train_preprocess,
        'dev': dev_preprocess,
        'test': test_preprocess
    }, output_path)
