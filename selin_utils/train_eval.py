"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The module that contains utility functions related to model training and evaluation
"""
import time
import logging
from typing import Dict, List, Tuple
from collections import Counter

from torch.optim import AdamW
# from bitsandbytes.optim import Adam8bit

from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import numpy as np
import json
import copy

from .dataset import get_class_names, get_label_column
from .dataset import get_tabular_column_type, load_tabular_info
from .dataset import get_single_input, get_single_context
from .utils import check_kwargs

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(
        self
    ):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self,
        val: float,
        n: int=1
    ):
        """
        Update the values
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class LossFunc():
    def __init__(
        self,
        model_name,
        true_matrix=None,
        train_y=None,
        num_classes=None,
        num_data=None,
        reg_lambda=1.0,
    ):
        self.nll_loss = nn.NLLLoss()
        self.mse_loss = nn.MSELoss()
        
        if model_name not in ['base', 'counselor']:
            raise ValueError(f'Model {model_name} is not supported')
        self.model_name = model_name
        self.reg_lambda = reg_lambda
        
        if self.model_name != 'base':
            assert(true_matrix != None)
            assert(train_y != None)
            assert(num_classes != None)
            assert(num_data != None)
            
            self.true_matrix = true_matrix
            self.train_y = train_y.long()
            self.num_data = num_data
            self.num_classes = num_classes
        
    def __call__(
        self,
        *inputs,
        consequent_estimation=False
    ):
        if consequent_estimation:
            assert(self.model_name != 'base')
            mu, alpha_x, alpha_w, coverage = inputs
            
            bsz, n_head, antecedent_len, num_atom = alpha_x.shape
            
            alpha_x = alpha_x.flatten(start_dim=0, end_dim=1)
            alpha_w = alpha_w.flatten(start_dim=0, end_dim=1)
            mu = mu.flatten(start_dim=0, end_dim=1)
            coverage = coverage.flatten(start_dim=0, end_dim=1)

            satis = torch.matmul(alpha_x, self.true_matrix)
            satis_num = torch.sum(satis, dim=1)
            satis_mu_mask = (satis_num == antecedent_len)
            
            mu_ = []
            for m in satis_mu_mask:
                satis_ans = self.train_y[m]
                satis_ans = F.one_hot(satis_ans.long(), num_classes=self.num_classes).float()
                mu_.append(torch.mean(satis_ans, dim=0))
            mu_ = torch.stack(mu_)
            mu_loss = F.mse_loss(mu, mu_)
            
            satis_regression_mask = (satis_num != 0)
            n = torch.sum(satis_regression_mask, dim=-1)
            coverage_ = n / self.num_data
            coverage_loss = F.mse_loss(coverage, coverage_)
                        
            regression_losses = []
            for i, b in enumerate(satis):
                m = satis_regression_mask[i]
                
                left = b.T[m]
                left = torch.mm(left, alpha_w[i])
                left = F.softmax(left, dim=-1)
                left = torch.log(left)

                right = self.train_y[m]
                regression_loss = self.nll_loss(left, right)
                    
                regression_losses.append(regression_loss)
                    
            regression_loss = torch.stack(regression_losses).mean()
            
            loss = mu_loss + coverage_loss + self.reg_lambda * regression_loss
        
        else:
            log_prob, label = inputs
            loss = self.nll_loss(log_prob, label)
            
        return loss
            
def train_epoch(
    optimizer: object,
    ce_optimizer: object,
    model: object,
    loss_func: object,
    train_dataloader: object,
    gpu: torch.device,
) -> Tuple[object, ...]:
    """
    Train the model for an epoch
    """
    train_losses = AverageMeter()
    ce_losses = AverageMeter()
    pbar = tqdm(train_dataloader)

    model.train()
    for batch in train_dataloader:
        inputs, y = batch
        inputs = [i.to(gpu) for i in inputs]
        y = y.to(gpu)
            
        bsz = len(y)

        if model.model_name == 'base':
            log_prob, _, _ = model(
                inputs
            )
        elif model.model_name == 'counselor':
            atom_prob_list = model.ag(
                inputs
            )
            
            mu, alpha_x, alpha_w, coverage = model.ce(
                atom_prob_list.detach()
            )
            
            ce_loss = loss_func(
                mu,
                alpha_x,
                alpha_w,
                coverage,
                consequent_estimation=True
            )
            
            ce_optimizer.zero_grad()
            ce_loss.backward()
            ce_optimizer.step()
            
            ce_losses.update(ce_loss.item(), n=bsz)
            
            mu, _, _, coverage = model.ce(
                atom_prob_list
            )
            
            # Categorical distribution probability
            n = coverage * loss_func.num_data
            sf = model.ag.beta * torch.reciprocal(n)
            sf = sf.unsqueeze(dim=-1).repeat(1, 1, loss_func.num_classes)
            prob = torch.div(mu + sf, 1 + loss_func.num_classes * sf)
            prob = torch.mean(prob, dim=1)
            
            log_prob = torch.log(prob)

        loss = loss_func(log_prob, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item(), n=bsz)

        pbar.update(1)
        desc = f'Train Loss: {train_losses.avg:.3f}'
        if model.model_name == 'counselor':
            desc += f', CE Loss: {ce_losses.avg:.3f}'
        pbar.set_description(desc)
                
    pbar.close()

    return model, train_losses

def eval_epoch(
    model: object,
    loss_func: object,
    valid_dataloader: object,
    class_names: List[str],
    gpu: torch.device,
) -> Tuple[dict, float, float, object]:
    """
    Evaluate the model for an epoch
    """
    pbar = tqdm(valid_dataloader)
    model.eval()
    with torch.inference_mode():
        valid_loss = AverageMeter()
        predictions = []
        target_probs = []
        answers = []
        for batch in valid_dataloader:
            inputs, y = batch
            inputs = [i.to(gpu) for i in inputs]
            y = y.to(gpu)
            batch_size = len(y)

            outputs, _, _ = model(
                inputs
            )

            loss = loss_func(outputs, y)
            valid_loss.update(loss.item(), n=batch_size)

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)

            target_prob = torch.exp(outputs)
            target_probs.extend(target_prob)

            answers.extend(y)

            pbar.update(1)
            pbar.set_description(f'Valid Loss: {valid_loss.avg:.3f}')
                        
        pbar.close()
        predictions = torch.stack(predictions).cpu().numpy()
        target_probs = torch.stack(target_probs).cpu().numpy()
        answers = torch.stack(answers).cpu().numpy()

        classification_dict = classification_report(
            answers,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        
        if len(class_names) > 2:
            ovo_roc_auc = roc_auc_score(answers, target_probs, multi_class='ovo')
            ovr_roc_auc = roc_auc_score(answers, target_probs, multi_class='ovr')
            
            roc_auc = (ovo_roc_auc, ovr_roc_auc)
            pr_auc = None
        else:
            roc_auc = roc_auc_score(answers, target_probs[:, 1])
            precision, recall, _ = precision_recall_curve(
                answers,
                predictions,
                pos_label=1
            )
            pr_auc = auc(recall, precision)

        return classification_dict, roc_auc, pr_auc, valid_loss

def train(
    model: object,
    loss_func: object,
    train_dataloader: object,
    valid_dataloader: object,
    gpu: torch.device,
    **kwargs
) -> object:
    """
    Train the model and evaluate for entire epochs with given train and valid dataloader.
    """
    check_kwargs(
        ['class_names', 'dir_path'],
        kwargs=kwargs
    )

    default_kwargs = {
        'learning_rate': 1e-5,
        'weight_decay': 0.0,
        'gamma': 0.95,
        'epochs': 10,
        'max_antecedent_len': 4,
    }
    default_kwargs.update(kwargs)
    kwargs = default_kwargs
    
    if model.model_name == 'base':
        main_params = model.parameters()
        ce_optimizer = None
    elif model.model_name == 'counselor':
        main_params = model.ag.parameters()
        ce_optimizer = AdamW(
            model.ce.parameters(),
            lr=kwargs['learning_rate'],
            weight_decay=kwargs['weight_decay']
        )
        ce_optimizer.zero_grad()
        
#     if model.base in []:
#         optimizer = Adam8bit(
#             main_params,
#             lr=kwargs['learning_rate'],
#             weight_decay=kwargs['weight_decay']
#         )
#     else:
    optimizer = AdamW(
        main_params,
        lr=kwargs['learning_rate'],
        weight_decay=kwargs['weight_decay']
    )
    optimizer.zero_grad()

    scheduler = ExponentialLR(optimizer, gamma=kwargs['gamma'])
    
    min_valid_loss = 100.0
    if model.base in ['llama2']:
        tf_best_model_path = kwargs['dir_path'] / 'model_best_tf'
    
    best_model_path = kwargs['dir_path'] / 'model_best.pt'

    train_times = AverageMeter()
    valid_times = AverageMeter()

    train_log = logging.getLogger()
    train_log.setLevel(logging.INFO)
    train_file_handler = logging.FileHandler(str(kwargs['dir_path'] / 'log'), mode='w')
    train_file_handler.setFormatter(logging.Formatter('%(message)s'))
    train_log.addHandler(train_file_handler)

    train_log.info('Start Training\n')
    for epoch in range(kwargs['epochs']):
        train_log.info(f'Epoch {epoch}')
        print(f'Epoch {epoch}')

        train_start = time.time()
        model, train_loss = train_epoch(
            optimizer,
            ce_optimizer,
            model,
            loss_func,
            train_dataloader,
            gpu,
        )
        train_end = time.time()
        train_time = train_end - train_start
        train_log.info(f'Training Time: {train_time:.3f} s')
        train_times.update(train_time)

        scheduler.step()

        valid_start = time.time()
        classification_dict, roc_auc, pr_auc, valid_loss = eval_epoch(
            model,
            loss_func,
            valid_dataloader,
            kwargs['class_names'],
            gpu,
        )
        valid_end = time.time()
        valid_time = valid_end - valid_start
        train_log.info(f'Validation Time: {train_time:.3f} s')
        valid_times.update(valid_time)

        valid_f1 = classification_dict["macro avg"]["f1-score"]
        train_log.info(f'Train Loss: {train_loss.avg:.3f}')
        train_log.info(f'Valid Loss: {valid_loss.avg:.3f}')
        train_log.info(f'Valid Macro-F1: {valid_f1:.4f}')
        if len(kwargs['class_names']) > 2:
            train_log.info(f'Valid OVO ROC-AUC: {roc_auc[0]:.4f}')
            train_log.info(f'Valid OVR ROC-AUC: {roc_auc[1]:.4f}')
        else:
            train_log.info(f'Valid ROC-AUC: {roc_auc:.4f}')
            train_log.info(f'Valid PR-AUC: {pr_auc:.4f}')
        train_log.info('\n')

#         model_path = kwargs['dir_path'] / f'model_epoch_{epoch}.pt'
#         torch.save(model.state_dict(), str(model_path))

        if valid_loss.avg < min_valid_loss:
            min_valid_loss = valid_loss.avg
            
            if model.base in ['llama2']:
                if model.model_name == 'base':
                    model.tf_model.save_pretrained(tf_best_model_path)
                    cur_tf_model = model.tf_model
                    model.tf_model = None
                    torch.save(model.state_dict(), str(best_model_path))
                    model.tf_model = cur_tf_model
                elif model.model_name == 'counselor':
                    model.ag.tf_model.save_pretrained(tf_best_model_path)
                    cur_tf_model = model.ag.tf_model
                    model.ag.tf_model = None
                    torch.save(model.state_dict(), str(best_model_path))
                    model.ag.tf_model = cur_tf_model
                else:
                    assert(0)
                
                
            else:
                torch.save(model.state_dict(), str(best_model_path))

    train_log.info(f'Average Train Time: {train_times.avg:.3f} s')
    train_log.info(f'Average Valid Time: {valid_times.avg:.3f} s')

    train_log.removeHandler(train_file_handler)
    train_file_handler.close()

    return model

def eval_model(
    model: object,
    loss_func: object,
    test_dataloader: object,
    true_matrix: torch.Tensor,
    gpu: torch.device,
    **kwargs
):
    """
    Evaluate the model with test dataloader.
    This function gives a more info compared to eval_epoch.
    """
    check_kwargs(
        ['class_names', 'dir_path'],
        kwargs=kwargs
    )
    
    eval_log = logging.getLogger()
    eval_log.setLevel(logging.INFO)
    eval_file_handler = logging.FileHandler(str(kwargs['dir_path'] / 'model_eval'), mode='w')
    eval_file_handler.setFormatter(logging.Formatter('%(message)s'))
    eval_log.addHandler(eval_file_handler)

    pbar = tqdm(test_dataloader)
    model.eval()
    with torch.inference_mode():
        test_loss = AverageMeter()
        predictions = []
        target_probs = []
        answers = []

        eval_start = time.time()
        for batch in test_dataloader:
            inputs, y = batch
            if model.base in ['llama2']:
                inputs = [i.to("cuda") for i in inputs]
            else:
                inputs = [i.to(gpu) for i in inputs]
            y = y.to(gpu)
            batch_size = len(y)

            # For base, we do not use atom_prob_list and cp_list
            outputs, atom_prob_list, cp_list = model(
                inputs
            )

            loss = loss_func(outputs,y)
            test_loss.update(loss.item(), batch_size)

            _, preds = torch.max(outputs, dim=1)
            target_prob = torch.exp(outputs)
            target_probs.extend(target_prob)

            predictions.extend(preds)
            answers.extend(y)

            pbar.update(1)
            pbar.set_description(f'Test Loss: {test_loss.avg:.3f}')

        pbar.close()
        eval_end = time.time()
        eval_time = eval_end - eval_start

        predictions = torch.stack(predictions).cpu().numpy()
        target_probs = torch.stack(target_probs).cpu().numpy()
        answers = torch.stack(answers).cpu().numpy()
        
        prediction_label = []
        for prob, answer in zip(target_probs, answers):
            prediction_label.append([prob.tolist(), int(answer)])

        prediction_label_path = kwargs['dir_path'] / 'eval_prediction_label.json'
        with open(prediction_label_path, 'w') as f:
            json.dump(prediction_label, f)

        if model.model_name == 'selor':
            count_length = Counter(torch.stack(lengths).int().tolist())
            dist_length = {}
            for length, count in count_length.items():
                dist_length[length] = count / len(answers)

        eval_log.info(f'Avg Test Loss: {test_loss.avg:.4f}')
        eval_log.info(f'Evaluation Time: {eval_time:.3f} s')

        if model.model_name == 'selor':
            eval_log.info(f'Confidence: {confidences.avg:.4f}')
            eval_log.info(f'Consistency: {consistencies.avg:.4f}')
            eval_log.info(f'Duplicate: {duplicates.avg:.4f}')
            eval_log.info(f'Unique: {uniques.avg:.4f}')
            eval_log.info(f'Coverage: {coverages.avg:.4f}')
            eval_log.info('Length')
            for i in range(max_antecedent_len + 1):
                if i in dist_length:
                    eval_log.info(f'{i}: {dist_length[i]:.4f}')

        if len(kwargs['class_names']) > 2:
            ovo_roc_auc = roc_auc_score(answers, target_probs, multi_class='ovo')
            ovr_roc_auc = roc_auc_score(answers, target_probs, multi_class='ovr')
            
            roc_auc = (ovo_roc_auc, ovr_roc_auc)
            pr_auc = None
        else:
            roc_auc = roc_auc_score(answers, target_probs[:, 1])
            precision, recall, _ = precision_recall_curve(
                answers,
                predictions,
                pos_label=1
            )
            pr_auc = auc(recall, precision)

        eval_log.info('Prediction Performance:')
        c_report = classification_report(
            answers,
            predictions,
            target_names=kwargs['class_names'],
            digits=4,
            zero_division=0,
        )
        eval_log.info(f'{c_report}')
        if len(kwargs['class_names']) > 2:
            eval_log.info(f'OVO ROC-AUC: {roc_auc[0]:.4f}')
            eval_log.info(f'OVR ROC-AUC: {roc_auc[1]:.4f}')
        else:
            eval_log.info(f'ROC-AUC: {roc_auc:.4f}')
            eval_log.info(f'PR-AUC: {pr_auc:.4f}')

        eval_log.removeHandler(eval_file_handler)
        eval_file_handler.close()

def get_explanation(
    model: object,
    true_matrix: torch.Tensor,
    atom_pool: object,
    inputs: Tuple[torch.Tensor, ...],
    class_names: List[str],
    gpu: torch.device,
) -> Tuple[Dict[str, float], List[str], List[float]]:
    """
    Get an explanation for the instance.
    """
    model.eval()

    with torch.inference_mode():
        inputs = [i.to(gpu).unsqueeze(dim=0) for i in inputs]
        
        atom_prob_list = model.ag(
                inputs
        )
        
        bsz, n_head, antecedent_len, n_atoms = atom_prob_list.shape
        
        mu, alpha_x, alpha_w, coverage = model.ce(
            atom_prob_list
        )
        
        n = coverage * model.num_data
        sf = model.ag.beta * torch.reciprocal(n)
        sf = sf.unsqueeze(dim=-1).repeat(1, 1, model.num_classes)
        cp_list = torch.div(mu + sf, 1 + model.num_classes * sf)
        cp = torch.mean(cp_list, dim=1).squeeze(dim=0)

        atom_prob_list = atom_prob_list.flatten(start_dim=0, end_dim=1)
        
        _, ind = torch.max(atom_prob_list, dim=-1)
        ind = ind.squeeze(dim=0)
        antecedents = [atom_pool.atoms[atom_pool.atom_id2key[i]] for i in ind]

        cover_antecedent_prob = torch.sum(atom_prob_list, dim=1)
        cover_antecedent_prob = torch.matmul(cover_antecedent_prob, true_matrix)
        mat_satis = (cover_antecedent_prob == model.ag.antecedent_len)
        mat_satis = torch.sum(mat_satis.float(), dim=-1)
        coverage = mat_satis / model.num_data
        coverage = coverage.item()
        
        class_probs = {}
        for i, _ in enumerate(cp):
            class_probs[f'{class_names[i]}'] = round(cp[i].item(), 4)

        antecedents = ' & '.join([atom.display_str for atom in antecedents])
        alpha_w = alpha_w.squeeze(dim=0).squeeze(dim=0).cpu().tolist()
        
        return class_probs, antecedents, coverage, alpha_w

def get_all_explanation(
    model: object,
    dataset: str,
    test_data: object,
    gpu: torch.device,
    **kwargs
) -> Tuple[List[str], List[dict]]:
    """
    Extract all explanations of given test dataset
    """
    check_kwargs(
        ['atom_pool', 'true_matrix'],
        kwargs=kwargs
    )
    atom_pool = kwargs['atom_pool']
    true_matrix = kwargs['true_matrix']

    class_names = get_class_names(dataset)
    y_col = get_label_column(dataset)

    if dataset in ['yelp', 'clickbait']:
        check_kwargs(
            ['tf_tokenizer', 'atom_tokenizer'],
            value=True,
            kwargs=kwargs
        )
        tf_tokenizer = kwargs['tf_tokenizer']
        atom_tokenizer = kwargs['atom_tokenizer']
        test_df = test_data

        tabular_column_type = None
        tabular_info = None

    elif dataset in ['adult', 'diabetes', 'credit']:
        tf_tokenizer = None
        atom_tokenizer = None
        test_df = test_data

        tabular_column_type = get_tabular_column_type(
            dataset=dataset
        )
        tabular_info = load_tabular_info(
            dataset=dataset
        )

    else:
        raise ValueError(f'Dataset {dataset} is not supported')
        
    exp_list = []
    result_list = []
    for target_id in tqdm(range(len(test_df)), desc='Extracting Explanations'):
        exp = ''
        row = test_df.iloc[target_id,:]
        
        target_context = get_single_context(
            row,
            dataset,
            tabular_column_type,
            tabular_info
        )

        inputs = get_single_input(
            row,
            dataset,
            atom_pool,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
        )
        
        exp += f'{target_context}\n'

        class_probs, antecedents, coverage, alpha_w = get_explanation(
            model,
            true_matrix,
            atom_pool,
            inputs,
            class_names,
            gpu
        )
        pred = max(class_probs, key=class_probs.get)

        y = int(row[y_col])

        label = class_names[y]

        exp += f'Label: {label}\n'
        exp += f'Prediction: {pred}\n\n'
        exp += 'Class Probability\n'
        for class_name, prob in class_probs.items():
            exp += f'{class_name}: {prob}\n'

        exp += '\n'
        exp += f'Explanation: {antecedents}\n'
        exp += f'Coverage: {coverage:.6f}\n'
        exp += f'alpha_w:\n{np.round(np.array(alpha_w), decimals=4)}\n'
        exp += '\n'
        
        result_dict = {
            'Id': target_id,
            'Target': target_context,
            'Label': label,
            'Prediction': pred,
            'Explanation': antecedents,
            'Class Probability': class_probs,
            'Coverage': coverage,
            'alpha_w': alpha_w,
        }

        result_list.append(result_dict)
        exp_list.append(exp)

    return exp_list, result_list

def get_base_embedding(
    model: object,
    train_dataloader: object,
    gpu: torch.device,
) -> torch.Tensor:
    """
    Get embedding of given base model and train dataloader.
    """
    assert model.model_name == 'base'

    h_list = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(train_dataloader):
            inputs, y = batch
            inputs = [i.to(gpu) for i in inputs]
            y = y.to(gpu)

            _, h, _ = model(
                inputs
            )

            h_list.append(h.cpu())
    embeddings = torch.cat(h_list, dim=0)
    return embeddings
