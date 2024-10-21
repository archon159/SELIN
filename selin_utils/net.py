"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The module that contains utility functions and classes for models
"""
from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import logging

import torchvision.models as vis_models

from .utils import check_kwargs
from .dataset import get_base_type, get_tabular_column_type

def get_tf_model(
    base: str='bert'
) -> Tuple[object, ...]:
    """
    Get transformer model for NLP task.
    """
    logging.set_verbosity_error()
    if base == 'bert':
        from transformers import BertModel, BertTokenizer, BertConfig
        pre_trained_model_name = 'bert-base-uncased'
        tf_tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
        tf_model = BertModel.from_pretrained(pre_trained_model_name, return_dict=True)
        config = BertConfig.from_pretrained(pre_trained_model_name)

    elif base == 'roberta':
        from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
        pre_trained_model_name = 'roberta-base'
        tf_tokenizer = RobertaTokenizer.from_pretrained(pre_trained_model_name)
        tf_model = RobertaModel.from_pretrained(pre_trained_model_name, return_dict=True)
        config = RobertaConfig.from_pretrained(pre_trained_model_name)

    elif base == 'llama2':
        from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
        from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        
        pre_trained_model_name = 'llama2'
        checkpoint_path = "/home/data/llama2/Llama-2-7b-hf"
        tf_tokenizer = LlamaTokenizer.from_pretrained(checkpoint_path)
        tf_tokenizer.pad_token = tf_tokenizer.eos_token
        
        config = LlamaConfig.from_pretrained(checkpoint_path)
        tf_model = LlamaModel.from_pretrained(checkpoint_path, device_map="auto")
        tf_model = prepare_model_for_kbit_training(tf_model, use_gradient_checkpointing=True)
        LORA_R = 8
        LORA_ALPHA = 16
        LORA_DROPOUT= 0.05
        LORA_TARGET_MODULES = [
            "q_proj",
            "v_proj",
        ]

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        tf_model = get_peft_model(tf_model, lora_config)
        
    else:
        tf_tokenizer = None
        tf_model = None
        config = None

    return tf_tokenizer, tf_model, config
    
class FTTransformer(nn.Module):
    def __init__(
        self,
        dataset: str='adult',
        embedding_dim: int=32,
        hidden_dim: int=32,
        input_dim: int=32,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
#         n_cat_options = get_tabular_num_cat_options(dataset)
        
        self.n_cat = len(categorical_x_col)
        self.n_num = len(numerical_x_col)
        
        n_cat_options = input_dim - self.n_num

        self.cls = nn.Parameter(torch.ones(1, embedding_dim), requires_grad=True)
        
        self.cat_w = nn.Embedding(
            num_embeddings=n_cat_options,
            embedding_dim=embedding_dim,
        )
        self.cat_b = nn.Parameter(torch.randn(self.n_cat, embedding_dim), requires_grad=True)
        
        self.num_w = nn.Parameter(torch.randn(self.n_num, embedding_dim), requires_grad=True)
        self.num_b = nn.Parameter(torch.randn(self.n_num, embedding_dim), requires_grad=True)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        self.te_layer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, x):
        bsz, n_columns = x.shape
        
        x_num = x[:, :self.n_num]
        x_cat = x[:, self.n_num:].int()
        
        assert(torch.all(torch.sum(x_cat, dim=-1) == self.n_cat))
        
        cls = self.cls.unsqueeze(0).tile(bsz, 1, 1)
        
        x_cat_index = [b.nonzero(as_tuple=True)[0] for b in x_cat]
        x_cat_index = torch.stack(x_cat_index, dim=0)
        
        cat_emb = self.cat_w(x_cat_index) + self.cat_b.unsqueeze(dim=0).tile(bsz, 1, 1)
        num_emb = torch.mul(
            x_num.unsqueeze(dim=-1).tile(1, 1, self.embedding_dim),
            self.num_w.unsqueeze(dim=0).tile(bsz, 1, 1)
        ) +  self.num_b.unsqueeze(dim=0).tile(bsz, 1, 1)
        
        emb = torch.cat((cls, cat_emb, num_emb), dim=1)
        
        out = self.te_layer(emb)
        out = out[:, 0, :]
        
        return out
    
class BaseModel(nn.Module):
    """
    The data structure for base models.
    NLP: Finetune transformer based models with 1-layer dnn.
    Tabular: 3-layer DNN with RELU activations.
    """
    def __init__(
        self,
        dataset: str='yelp',
        base: str='bert',
        fix_backbone: bool=False,
        **kwargs
    ):
        super().__init__()
        default_kwargs = {
            'input_dim': 512,
            'hidden_dim': 768,
            'embedding_dim': 768,
            'num_classes': 2,
        }
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        self.model_name = 'base'

        self.dataset = dataset
        self.base = base
        self.btype = get_base_type(base)
        
        self.fix_backbone = fix_backbone

        hidden_dim = kwargs['hidden_dim']
        embedding_dim = kwargs['embedding_dim']
        input_dim = kwargs['input_dim']
        num_classes = kwargs['num_classes']

        if self.btype == 'tab':
            if base == 'dnn':
                self.tab_base = nn.Sequential(
                    nn.Linear(input_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, hidden_dim),
                )
            elif base == 'fttransformer':
                self.tab_base = FTTransformer(
                    dataset=dataset,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    input_dim=input_dim,
                )
            else:
                assert(0)
            
        elif self.btype == 'nlp':
            self.tf_model = kwargs['tf_model']
        else:
            assert(0)

        self.base_head = nn.Linear(hidden_dim, num_classes)

    def get_base_embedding(
        self,
        inputs: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if self.btype == 'tab':
            x, x_ = inputs
            h = self.tab_base(x)

        elif self.btype == 'nlp':
            input_ids, attention_mask, x_ = inputs
            out = self.tf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            full_embedding = out['last_hidden_state']
            if self.base == 'llama2':
                batch_size = input_ids.shape[0]
                sequence_lengths = (
                    torch.ne(
                        input_ids,
                        self.tf_model.config.pad_token_id
                    ).sum(dim=-1) - 1
                )
                h = full_embedding[torch.arange(batch_size, device=full_embedding.device), sequence_lengths]
            else:
                h = full_embedding[:,0]
        else:
            assert(0)
            
        return h, x_
        
    def forward(
        self,
        inputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Base model forward propagation
        """
        if self.fix_backbone:
            with torch.no_grad():
                h, _ = self.get_base_embedding(inputs)
        else:
            h, _ = self.get_base_embedding(inputs)

        out = self.base_head(h)
        out = nn.functional.softmax(out, dim=1)
        out = torch.log(out)

        return out, h, _

class ConsequentEstimator(nn.Module):
    """
    The data structure for the consequent estimator.
    Predict mu, sigma, and coverage of given antecedent with 3-layer dnns.
    """
    def __init__(
        self,
        atom_embedding: torch.Tensor=None,
        embedding_dim: int=768,
        hidden_dim: int=768,
        num_classes: int=2,
    ):
        super().__init__()
        self.num_classes = num_classes
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.cp_te = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.atom_embedding=atom_embedding

        self.w_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_classes)
        )
        
        self.c_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(
        self,
        alpha_x,
    ):
        bsz, n_head, antecedent_len, n_atoms = alpha_x.shape
        
        x_list = []
        w_list = []
        
        # Flatten batch dimension and head dimension
        # alpha_x -> (B*H, L, N_a)
        alpha_x = alpha_x.flatten(start_dim=0, end_dim=1)
        dummy_mask = (alpha_x[:, :, 0] == 1)
        
        # alpha_x_ -> (B*H, L, D)
        alpha_x_ = torch.matmul(alpha_x, self.atom_embedding)
        alpha_x_ = self.cp_te(alpha_x_)
        # alpha_w -> (B*H, L, N_c)
        alpha_w = self.w_head(alpha_x_)
        alpha_w[dummy_mask] = 0.
        
        coverage = torch.sigmoid(self.c_head(torch.mean(alpha_x_, dim=1)))
        
        out = torch.sum(alpha_w, dim=1)
        mu = F.softmax(out, dim=-1)
        
        mu = mu.reshape(bsz, n_head, self.num_classes)
        alpha_x = alpha_x.reshape(bsz, n_head, antecedent_len, n_atoms)
        alpha_w = alpha_w.reshape(bsz, n_head, antecedent_len, self.num_classes)
        coverage = coverage.reshape(bsz, n_head)

        return mu, alpha_x, alpha_w, coverage

class AtomSelector(nn.Module):
    """
    The data structure for the atom selector.
    Choose an atom for given instance.
    """
    def __init__(
        self,
        num_atoms: int=154,
        antecedent_len: int=2,
        hidden_dim: int=768,
        embedding_dim: int=768,
        atom_embedding: torch.Tensor=None,
        avoid_dummy: bool=False
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=embedding_dim,
        )
        self.dropout = nn.Dropout(0.1)
        self.gru_head = nn.Linear(embedding_dim, num_atoms)

        self.embedding_dim = embedding_dim
        self.antecedent_len = antecedent_len
        self.atom_embedding = atom_embedding
        self.avoid_dummy = avoid_dummy
        
        self.zero_v = None

    def filtered_softmax(
        self,
        x: torch.Tensor,
        x_: torch.Tensor,
        pos: int,
        pre_max_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conduct Gumbel-softmax for atoms that is satisfied by the instance.
        If an instance does not satisfy any atom, then choosen NULL atom.
        If NULL atom is once chosen, then the following atoms also become NULL atom.
        """
        assert len(x) == len(x_)
        if pos == 0:
            x_[torch.sum(x_, dim=-1) == 0, 0] = 1.0
        else:
            assert pre_max_index != None
            
            if self.avoid_dummy:
                # Previous: Dummy
                x_[(pre_max_index==0), 1:] = 0.
                x_[(pre_max_index==0), 0] = 1.

                # Normal case
                x_[(pre_max_index!=0), 0] = 0.
                
                # Nothing but Dummy
                x_sum = torch.sum(x_, dim=-1)
                only_dummy_idx = (x_sum == 0.)
                x_[only_dummy_idx, 0] = 1.
            else:
                x_[(pre_max_index==0), :] = 0.
                x_[:, 0] = 1.0

        x[torch.logical_not(x_)] = float('-inf')
        if self.training:
            x = F.gumbel_softmax(logits=x, tau=1, hard=True, dim=1)
        else:
            bsz, n_atom = x.shape
            _, x = torch.max(x, dim=-1)
            x = F.one_hot(x, num_classes=n_atom).float()

        return x

    def forward(
        self,
        representation_emb: torch.Tensor,
        x_: torch.Tensor,
    ) -> torch.Tensor:
        """
        Antecedent generator forward propagation
        """
        representation_emb = representation_emb.unsqueeze(dim=0).contiguous()
        cur_input = representation_emb
        cur_h_0 = None

        atom_prob = []
        if self.zero_v is None:
            self.zero_v = torch.zeros(x_.shape).to(x_.device).long().detach()

        max_index = None
        for j in range(self.antecedent_len):
            if cur_h_0 != None:
                _, h_n = self.gru(cur_input, cur_h_0)
            else:
                _, h_n = self.gru(cur_input)

            cur_h_0 = h_n
            h_n = h_n.squeeze(dim=0)
            h_n = self.dropout(h_n)
            out = self.gru_head(h_n)

            prob = self.filtered_softmax(out, x_, j, max_index)

            _, ind = torch.max(prob, dim=-1)
            ind = ind.unsqueeze(dim=1)
            src = self.zero_v
            x_ = torch.scatter(x_, dim=1, index=ind, src=src)

            atom_prob.append(prob)
            max_index = torch.max(prob, dim=-1)[1]

            atom_wsum = torch.mm(prob, self.atom_embedding.detach())
            cur_input = representation_emb + atom_wsum.unsqueeze(dim=0)

        atom_prob = torch.stack(atom_prob, dim=1)

        return atom_prob


class AntecedentGenerator(BaseModel):
    """
    Data structure for antecedent generator.
    Sequentially chooses atoms with atom selectors,
    and obtain consequent by the consequent estimator.
    """
    def __init__(
        self,
        dataset: str='yelp',
        base: str='bert',
        atom_embedding: torch.Tensor=None,
        num_data: int=56000,
        **kwargs
    ):
        default_kwargs = {
            'antecedent_len': 4,
            'head': 1,
            'num_atoms': 5001,
            'input_dim': 512,
            'hidden_dim': 768,
            'embedding_dim': 768,
            'num_classes': 2,
            'tf_model': None,
            'avoid_dummy': False,
            'fix_backbone': False,
        }
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        super().__init__(
            dataset=dataset,
            base=base,
            input_dim=kwargs['input_dim'],
            hidden_dim=kwargs['hidden_dim'],
            embedding_dim=kwargs['embedding_dim'],
            tf_model=kwargs['tf_model'],
            num_classes=kwargs['num_classes'],
        )

        self.rs_list = nn.ModuleList([
            AtomSelector(
                num_atoms=kwargs['num_atoms'],
                antecedent_len=kwargs['antecedent_len'],
                hidden_dim=kwargs['hidden_dim'],
                embedding_dim=kwargs['embedding_dim'],
                atom_embedding=atom_embedding,
                avoid_dummy=kwargs['avoid_dummy'],
            ) for i in range(kwargs['head'])
        ])

        self.head = kwargs['head']
        self.antecedent_len = kwargs['antecedent_len']

        self.num_classes = kwargs['num_classes']
        self.num_atoms = kwargs['num_atoms']
        self.num_data = num_data

        self.base = base
        self.dataset = dataset

        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.atom_embedding = nn.Embedding(
            kwargs['num_atoms'],
            kwargs['hidden_dim'],
            _weight=atom_embedding
        )
        
        self.fix_backbone = kwargs['fix_backbone']
        
    def forward(
        self,
        inputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        if self.fix_backbone:
            with torch.no_grad():
                h, x_ = self.get_base_embedding(inputs)
        else:
            h, x_ = self.get_base_embedding(inputs)

        atom_prob_list = []
        class_prob_list = []
        for i in range(self.head):
            # Choose atoms
            atom_prob = self.rs_list[i](h, x_.clone().detach())
            atom_prob_list.append(atom_prob)

        atom_prob_list = torch.stack(atom_prob_list, dim=1)
        
        return atom_prob_list

class CounSelor(nn.Module):
    def __init__(
        self,
        dataset: str='yelp',
        base: str='bert',
        atom_embedding: torch.Tensor=None,
        **kwargs
    ):
        default_kwargs = {
            'antecedent_len': 4,
            'head': 1,
            'num_atoms': 5001,
            'input_dim': 512,
            'hidden_dim': 768,
            'embedding_dim': 768,
            'num_classes': 2,
            'num_data': 56000,
            'tf_model': None,
            'avoid_dummy': False,
            'fix_backbone': False,
        }
        default_kwargs.update(kwargs)
        kwargs = default_kwargs
        super().__init__()
        
        self.model_name = 'counselor'
        
        self.dataset = dataset
        self.base = base
        
        self.ag = AntecedentGenerator(
            dataset=dataset,
            base=base,
            atom_embedding=atom_embedding,
            antecedent_len=kwargs['antecedent_len'],
            head=kwargs['head'],
            num_atoms=kwargs['num_atoms'],
            input_dim=kwargs['input_dim'],
            hidden_dim=kwargs['hidden_dim'],
            embedding_dim=kwargs['embedding_dim'],
            num_classes=kwargs['num_classes'],
            num_data=kwargs['num_data'],
            tf_model=kwargs['tf_model'],
            avoid_dummy=kwargs['avoid_dummy'],
            fix_backbone=kwargs['fix_backbone'],
        )
        
        self.ce = ConsequentEstimator(
            atom_embedding=atom_embedding,
            hidden_dim=kwargs['hidden_dim'],
            embedding_dim=kwargs['embedding_dim'],
            num_classes=kwargs['num_classes'],
        )
        
        self.num_data = kwargs['num_data']
        self.num_classes = kwargs['num_classes']
        
    def forward(
        self,
        inputs,
        consquent_estimation=False
    ):
        atom_prob_list = self.ag(
            inputs
        )

        mu, alpha_x, alpha_w, coverage = self.ce(
            atom_prob_list
        )
        
        n = coverage * self.num_data
        sf = self.ag.beta * torch.reciprocal(n)
        sf = sf.unsqueeze(dim=-1).repeat(1, 1, self.num_classes)
        prob = torch.div(mu + sf, 1 + self.num_classes * sf)
        log_prob = torch.log(torch.mean(prob, dim=1))
        
        return log_prob, atom_prob_list, prob
