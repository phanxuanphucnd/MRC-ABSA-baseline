# -*- coding: utf-8 -*-

import os
import torch
import pickle
import argparse
from tqdm import  tqdm
from dataset import DualSample

def get_text(lines):
    text_list = []
    aspect_list = []
    for line in lines:
        # temp = line.split('####')
        # assert len(temp) == 3

        word_list = line.split()
        # aspect_label_list = [t.split('=')[-1] for t in temp[1].split()]
        # assert len(word_list) == len(aspect_label_list)

        text_list.append(word_list)
        # aspect_list.append(aspect_label_list)

    return text_list, aspect_list


def valid_data(triplet, aspect):
    for t in triplet[0][0]:
        assert aspect[t] != ['O']


def fusion_dual_pair(triplet):
    pair_aspect = []
    pair_sentiment = []

    for t in triplet:
        if t[0] not in pair_aspect:
            pair_aspect.append(t[0])
            pair_sentiment.append(t[2])

    return pair_aspect, pair_sentiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='./data/14rest',
                        help="Path to the dataset.")
    parser.add_argument('--output_path', type=str, default='./data/14rest/preprocess',
                        help='Path to the saved data.')

    args = parser.parse_args()

    DATASET_TYPE_LIST = ['train', 'dev', 'test']

    for dataset_type in DATASET_TYPE_LIST:
        #TODO: Read triple data
        with open(f'{args.data_path}/pair/{dataset_type}_pair.pkl', 'rb') as f:
            triple_data = pickle.load(f)

        #TODO: Read text
        with open(f'{args.data_path}/{dataset_type}.txt', 'r', encoding='utf-8') as f:
            text_lines = f.readlines()

        #TODO: Get text
        text_list, aspect_list = get_text(text_lines)

        sample_list = []
        header_fmt = 'Processing {:>5s}'
        for i in tqdm(range(len(text_list)), desc=f"{header_fmt.format(dataset_type.upper())}"):
            triplet = triple_data[i]
            text = text_list[i]
            #TODO: Valid data
            # valid_data(triplet, aspect_list[i])
            pair_aspect, pair_sentiment = fusion_dual_pair(triplet)

            forward_query_list = []
            forward_answer_list = []
            sentiment_query_list = []
            sentiment_answer_list = []

            forward_query_list.append("What aspects ?".split())
            start = [0]*len(text)
            end = [0]*len(text)
            for ta in pair_aspect:
                start[ta[0]] = 1
                end[ta[-1]] = 1
            forward_answer_list.append([start, end])

            for idx in range(len(pair_aspect)):
                ta = pair_aspect[idx]
                #TODO: Sentiment query
                query = f"What sentiment given the aspect {' '.join(text[ta[0]: ta[-1] + 1])} ?".split()
                sentiment_query_list.append(query)
                sentiment_answer_list.append(pair_sentiment[idx])

            sample = DualSample(
                text_lines[i],
                text,
                forward_query_list,
                forward_answer_list,
                sentiment_query_list,
                sentiment_answer_list
            )
            sample_list.append(sample)

        #TODO: Storages samples to .pt file
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        output_path = f"{args.output_path}/{dataset_type}_DUAL.pt"
        print(f"Saved data to `{output_path}`.")
        torch.save(sample_list, output_path)
