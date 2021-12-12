
import torch
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.CGBERT import *
from model.QACGBERT import *

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm, trange

from util.optimization import BERTAdam
from util.processor import (Sentihood_NLI_M_Processor,
                            Semeval_NLI_M_Processor)

from util.tokenization import *

from util.evaluation import *
from util.train_helper import getModelOptimizerTokenizer, convert_examples_to_features, system_setups
import logging

logger = logging.getLogger(__name__)


processors = {
    "sentihood_NLI_M":Sentihood_NLI_M_Processor,
    "semeval_NLI_M":Semeval_NLI_M_Processor
}

def data_and_model_loader(device, n_gpu, args, sampler="randomWeight"):

    processor = processors[args.task_name]()
    label_list = processor.get_labels()

    # model and optimizer
    model, _, tokenizer = \
        getModelOptimizerTokenizer(model_type=args.model_type,
                                   vocab_file=args.vocab_file,
                                   bert_config_file=args.bert_config_file,
                                   init_checkpoint=args.init_checkpoint,
                                   label_list=label_list,
                                   do_lower_case=True,
                                   num_train_steps=10,
                                   learning_rate=args.learning_rate,
                                   base_learning_rate=args.base_learning_rate,
                                   warmup_proportion=args.warmup_proportion)

    
    # test set
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length,
        tokenizer, args.max_context_length,
        args.context_standalone, args)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_seq_len = torch.tensor([[f.seq_len] for f in test_features], dtype=torch.long)
    all_context_ids = torch.tensor([f.context_ids for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_len, all_context_ids)
    test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    return model, test_dataloader

def evaluate(test_dataloader, model, device, n_gpu, nb_tr_steps, tr_loss, epoch, 
             global_step, output_log_file, global_best_acc, args):

    model.eval()
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    pbar = tqdm(test_dataloader, desc="Iteration")
    y_true, y_pred, score = [], [], []
    # we don't need gradient in this case.
    with torch.no_grad():
        for _, batch in enumerate(pbar):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # truncate to save space and computing resource
            input_ids, input_mask, segment_ids, label_ids, seq_lens, \
                context_ids = batch
            max_seq_lens = max(seq_lens)[0]
            input_ids = input_ids[:,:max_seq_lens]
            input_mask = input_mask[:,:max_seq_lens]
            segment_ids = segment_ids[:,:max_seq_lens]

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            seq_lens = seq_lens.to(device)
            context_ids = context_ids.to(device)

            # intentially with gradient
            tmp_test_loss, logits, _, _, _, _ = \
                model(input_ids, segment_ids, input_mask, seq_lens,
                        device=device, labels=label_ids,
                        context_ids=context_ids)

            logits = F.softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            tmp_test_accuracy=np.sum(outputs == label_ids)

            y_true.append(label_ids)
            y_pred.append(outputs)
            score.append(logits) #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_accuracy

            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples

    # we follow previous works in calculating the metrics
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    score = np.concatenate(score, axis=0)

    logger.info("***** Evaluation results *****")
    result = collections.OrderedDict()
    # handling corner case for a checkpoint start
    if nb_tr_steps == 0:
        loss_tr = 0.0
    else:
        loss_tr = tr_loss/nb_tr_steps

    # for ABSA tasks, we need more evaluations
    if args.task_name == "sentihood_NLI_M":
        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
        result = {'epoch': epoch,
                  'global_step': global_step,
                  'loss': loss_tr,
                  'test_loss': test_loss,
                  'test_accuracy': test_accuracy,
                  'aspect_strict_Acc': aspect_strict_Acc,
                  'aspect_Macro_F1': aspect_Macro_F1,
                  'aspect_Macro_AUC': aspect_Macro_AUC,
                  'sentiment_Acc': sentiment_Acc,
                  'sentiment_Macro_AUC': sentiment_Macro_AUC}
    else:
        aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
        sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
        sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
        sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
        result = {'epoch': epoch,
                  'global_step': global_step,
                  'loss': loss_tr,
                  'test_loss': test_loss,
                  'test_accuracy': test_accuracy,
                  'aspect_P': aspect_P,
                  'aspect_R': aspect_R,
                  'aspect_F': aspect_F,
                  'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
                  'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
                  'sentiment_Acc_2_classes': sentiment_Acc_2_classes}

    with open(output_log_file, "a+") as writer:
        for key in result.keys():
            logger.info("  %s = %s\n", key, str(result[key]))
            writer.write("%s\t" % (str(result[key])))
        writer.write("\n")

    # save for each time point
    if args.output_dir:
        torch.save(model.state_dict(), args.output_dir + "checkpoint.bin")
        if args.task_name == "sentihood_NLI_M":
            if aspect_strict_Acc > global_best_acc:
                torch.save(model.state_dict(), args.output_dir + "best_checkpoint.bin")
                global_best_acc = aspect_strict_Acc
        else:
            if aspect_F > global_best_acc:
                torch.save(model.state_dict(), args.output_dir + "best_checkpoint.bin")
                global_best_acc = aspect_F

    return global_best_acc

def run(args):

    device, n_gpu, _= system_setups(args)
    model, test_dataloader = data_and_model_loader(device, n_gpu, args)

if __name__ == "__main__":
    from util.args_parser import parser
    args = parser.parse_args()
    run(args)