# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import torch
import utils
import argparse
import numpy as np

from tqdm import tqdm
from model import BMRCBertModel
from torch.nn import functional as F
from dataset import BMRCDataset, generate_fi_batches
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer


#TODO: Init logger
logger = utils.get_logger('./logs.txt')

# seed = 123
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


def main(args, tokenizer):

    data_path = f"{args.data_path}/data.pt"
    standard_data_path = f"{args.data_path}/data_standard.pt"

    #TODO: Load data
    logger.info(f"Loading data...")
    total_data = torch.load(data_path)
    standard_data = torch.load(standard_data_path)

    train_data = total_data['train']
    dev_data = total_data['dev']
    test_data = total_data['test']
    dev_standard = standard_data['dev']
    test_standard = standard_data['test']

    #TODO: Init model
    logger.info(f"Initial model...")
    model = BMRCBertModel(args)
    if args.ifgpu:
        model = model.cuda()

    if args.mode == 'test':
        logger.info('Start testing...')
        test_dataset = BMRCDataset(train_data, dev_data, test_data, 'test')
        # load checkpoint
        logger.info(f'Loading model path: `{args.save_model_path}`.')
        checkpoint = torch.load(args.save_model_path)
        model.load_state_dict(checkpoint['net'])
        model.eval()

        batch_generator_test = generate_fi_batches(
            dataset=test_dataset, batch_size=1, shuffle=False, ifgpu=args.ifgpu)
        # eval
        logger.info('Evaluating...')
        f1 = test(args, model, tokenizer, batch_generator_test, test_standard, args.inference_beta)

    elif args.mode == 'train':
        train_dataset = BMRCDataset(train_data, dev_data, test_data, 'train')
        dev_dataset = BMRCDataset(train_data, dev_data, test_data, 'dev')
        test_dataset = BMRCDataset(train_data, dev_data, test_data, 'test')

        logger.info(f'------- Dataset Info -------')
        logger.info(f'Length of train dataset: {len(train_dataset)}')
        logger.info(f'Length of dev dataset  : {len(dev_dataset)}')
        logger.info(f'Length of test dataset : {len(test_dataset)}')
        logger.info(f'----------------------------')

        batch_num_train = train_dataset.get_batch_num(args.batch_size)

        # optimizer
        logger.info('Initial optimizer...')
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "_bert" in n], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if "_bert" not in n], 'lr': args.learning_rate, 'weight_decay': 0.01}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.tuning_bert_rate, correct_bias=False)

        # load saved model, optimizer and epoch num
        if args.reload and os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info('Reload model and optimizer after training epoch {}'.format(checkpoint['epoch']))
        else:
            start_epoch = 1
            logger.info('New model and optimizer from epoch 0')

        # scheduler
        training_steps = args.epoch_num * batch_num_train
        warmup_steps = int(training_steps * args.warm_up)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)

        # training
        logger.info('Begin training...')
        best_dev_f1 = 0.
        for epoch in tqdm(range(start_epoch, args.epoch_num + 1)):
            model.train()
            model.zero_grad()

            batch_generator = generate_fi_batches(
                train_dataset, batch_size=args.batch_size, ifgpu=args.ifgpu)

            for batch_index, batch_dict in enumerate(batch_generator):
                optimizer.zero_grad()
                # q1_a
                f_aspect_start_scores, f_aspect_end_scores = model(batch_dict['forward_asp_query'],
                                                                   batch_dict['forward_asp_query_mask'],
                                                                   batch_dict['forward_asp_query_seg'], 0)
                f_asp_loss = utils.calculate_entity_loss(
                    f_aspect_start_scores, f_aspect_end_scores,
                    batch_dict['forward_asp_answer_start'],
                    batch_dict['forward_asp_answer_end'])

                # q_3
                sentiment_scores = model(
                    batch_dict['sentiment_query'].view(-1, batch_dict['sentiment_query'].size(-1)),
                    batch_dict['sentiment_query_mask'].view(-1, batch_dict['sentiment_query_mask'].size(-1)),
                    batch_dict['sentiment_query_seg'].view(-1, batch_dict['sentiment_query_seg'].size(-1)), 1)
                sentiment_loss = utils.calculate_sentiment_loss(
                    sentiment_scores, batch_dict['sentiment_answer'].view(-1))

                # loss
                loss_sum = f_asp_loss + args.beta*sentiment_loss

                loss_sum.backward()
                optimizer.step()
                scheduler.step()

                # train logger
                if batch_index % 50 == 0:
                    logger.info(
                        'Epoch {}/{} - Batch {}/{}:\t Loss Sum:{}\t Forward Loss:{};{}\t Backward Loss:{};{}\t Sentiment Loss:{}'.
                            format(epoch, args.epoch_num, batch_index, batch_num_train,
                                   round(loss_sum.item(), 4), round(f_asp_loss.item(), 4),
                                   round(f_opi_loss.item(), 4), round(b_asp_loss.item(), 4),
                                   round(b_opi_loss.item(), 4), round(sentiment_loss.item(), 4))
                    )

            # validation
            batch_generator_dev = generate_fi_batches(
                dataset=dev_dataset, batch_size=1, shuffle=False, ifgpu=args.ifgpu)
            logger.info(f"Evaluate Dev...")
            f1 = test(args, model, tokenizer, batch_generator_dev, dev_standard, args.inference_beta)
            # save model and optimizer
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                logger.info('Model saved after epoch {}'.format(epoch))
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

                model_dir = '/'.join(args.save_model_path.split('/')[: -1])
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(state, args.save_model_path)

            # test
            batch_generator_test = generate_fi_batches(
                dataset=test_dataset, batch_size=1, shuffle=False, ifgpu=args.ifgpu)
            logger.info(f"Evaluate Test...")
            f1 = test(args, model, tokenizer, batch_generator_test, test_standard, args.inference_beta)

    else:
        logger.error('Error mode!')
        exit(1)
        
def test(args, model, tokenizer, batch_generator, standard, beta):
    model.eval()

    asp_target_num = 0
    asp_pol_target_num = 0

    asp_predict_num = 0
    asp_pol_predict_num = 0

    asp_match_num = 0
    asp_pol_match_num = 0

    for batch_index, batch_dict in enumerate(batch_generator):
        asp_target = standard[batch_index]['asp_target']
        asp_pol_target = standard[batch_index]['asp_pol_target']

        asp_predict = []
        asp_pol_predict = []

        final_asp_list = []
        final_asp_ind_list = []

        # forward q_1
        passenge_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero()
        passenge = batch_dict['forward_asp_query'][0][passenge_index].squeeze(1)

        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'], 0)

        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []

        for i in range(f_asp_start_ind.size(0)):
            if batch_dict['forward_asp_answer_start'][0, i] != -1:
                if f_asp_start_ind[i].item() == 1:
                    f_asp_start_index_temp.append(i)
                    f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
                if f_asp_end_ind[i].item() == 1:
                    f_asp_end_index_temp.append(i)
                    f_asp_end_prob_temp.append(f_asp_end_prob[i].item())

        f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)

        for idx in range(len(f_opi_start_index)):
            asp = [batch_dict['forward_asp_query'][0][j].item() for j in
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
            sentiment_query = torch.cat([sentiment_query, passenge], -1).unsqueeze(0)
            sentiment_query_seg += [1] * passenge.size(0)
            sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
            sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

            sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
            sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

            asp_f = []
            asp_f.append(final_asp_ind_list[idx][0])
            asp_f.append(final_asp_ind_list[idx][1])

            if asp_f + [sentiment_predicted] not in asp_pol_predict:
                asp_pol_predict.append(asp_f + [sentiment_predicted])
            if asp_f not in asp_predict:
                asp_predict.append(asp_f)

        asp_target_num += len(asp_target)
        asp_pol_target_num += len(asp_pol_target)

        asp_predict_num += len(asp_predict)
        asp_pol_predict_num += len(asp_pol_predict)

        for trip in asp_target:
            for trip_ in asp_predict:
                if trip_ == trip:
                    asp_match_num += 1

        for trip in asp_pol_target:
            for trip_ in asp_pol_predict:
                if trip_ == trip:
                    asp_pol_match_num += 1

    precision_aspect = float(asp_match_num) / float(asp_predict_num+1e-6)
    recall_aspect = float(asp_match_num) / float(asp_target_num+1e-6)
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect+1e-6)
    logger.info('Aspect - Precision: {} ; Recall: {} ; F1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

    precision_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_predict_num+1e-6)
    recall_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_target_num+1e-6)
    f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
            precision_aspect_sentiment + recall_aspect_sentiment+1e-6)
    logger.info('Aspect-Sentiment - Precision: {} ; Recall: {} ; F1: {}'.format(
        precision_aspect_sentiment, recall_aspect_sentiment, f1_aspect_sentiment))

    return f1_aspect_sentiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bidirectional MRC-based sentiment triplet extraction')

    parser.add_argument('--data_path', type=str, default="./data/14lap/preprocess/")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--reload', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="./model/14lap/model_final.pt")
    parser.add_argument('--save_model_path', type=str, default="./models/model.pt")
    parser.add_argument('--model_name', type=str, default="1")

    # model hyper-parameter
    parser.add_argument('--model_type', type=str, default="bert-base-uncased")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.8)

    # training hyper-parameter
    parser.add_argument('--ifgpu', type=bool, default=True)
    parser.add_argument('--epoch_num', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tuning_bert_rate', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_type)

    main(args, tokenizer)

