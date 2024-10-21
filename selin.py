"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The script to train selor
"""
from pathlib import Path
import json
from datetime import datetime
import torch
from torch import nn
import pandas as pd

# Import from custom files
from selin_utils import atom as at
from selin_utils import dataset as ds
from selin_utils import net
from selin_utils import train_eval as te
from selin_utils import utils

RUN = 'selin'

if __name__ == "__main__":
    args = utils.parse_arguments()

    dtype = ds.get_dataset_type(args.dataset)
    btype = ds.get_base_type(args.base)

    assert dtype == btype

    seed = args.seed
    
    gpu = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(gpu)

    utils.reset_seed(seed)

    tf_tokenizer, tf_model, config = net.get_tf_model(args.base)

    train_data, valid_data, test_data = ds.load_data(dataset=args.dataset)

    atom_pool_path = f'./{args.save_dir}/atom_pool/'
    atom_pool_path += f'atom_pool_{args.dataset}'
    if dtype == 'nlp':
        atom_tokenizer_path = f'./{args.save_dir}/atom_tokenizer/'
        atom_tokenizer_path += f'atom_tokenizer_{args.dataset}_seed_{args.seed}'
        atom_tokenizer = pd.read_pickle(atom_tokenizer_path)

        atom_pool_path += f'_num_atoms_{args.num_atoms}'
    elif args.dataset in ['adult', 'diabetes', 'credit']:
        atom_tokenizer = None
        if args.num2cat:
            atom_pool_path += '_num2cat'
    else:
        assert(0)
        
    atom_pool_path += f'_seed_{args.seed}'
    ap = pd.read_pickle(atom_pool_path)

    # Load class names
    class_names = ds.get_class_names(args.dataset)

    # Whether each train sample satisfies each atom
    true_matrix = at.get_true_matrix(ap)
    norm_true_matrix = true_matrix / (torch.sum(true_matrix, dim=1).unsqueeze(dim=1) + 1e-8)

    # Embedding from the base model for each train sample
    data_embedding_path = f'./{args.save_dir}/base_models/'
    data_embedding_path += f'base_{args.base}_dataset_{args.dataset}'
    if args.noise_ratio > 0:
        data_embedding_path += f'_noise_ratio_{args.noise_ratio}'
    data_embedding_path += f'_seed_{args.seed}'
    data_embedding_path = Path(data_embedding_path) / 'train_embeddings.pt'
    data_embedding = torch.load(data_embedding_path, map_location='cpu')

    # Obtain atom embedding
    atom_embedding = torch.mm(norm_true_matrix.to(gpu), data_embedding.to(gpu)).detach()

    # Create datasets
    train_dataset = ds.create_dataset(
        train_data,
        is_train=True,
        dataset=args.dataset,
        atom_pool=ap,
        atom_tokenizer=atom_tokenizer,
        tf_tokenizer=tf_tokenizer,
        config=config,
        noise_ratio=args.noise_ratio,
    )
    
    valid_dataset, test_dataset = [
        ds.create_dataset(
            data,
            is_train=False,
            dataset=args.dataset,
            atom_pool=ap,
            atom_tokenizer=atom_tokenizer,
            tf_tokenizer=tf_tokenizer,
            config=config,
        ) for data in [valid_data, test_data]]

    train_dataloader, valid_dataloader, test_dataloader = [
        ds.create_dataloader(
            dtset,
            args.batch_size,
            args.num_workers,
            shuffle=dtset.is_train,
        ) for dtset in [train_dataset, valid_dataset, test_dataset]]
    
    if dtype == 'nlp':
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
        embedding_dim = 768
    elif dtype == 'tab':
        input_dim = train_dataset.x.shape[1]
        hidden_dim = args.hidden_dim
        embedding_dim = 512
    else:
        raise ValueError(f'Dataset type {dtype} is not supported.')

    model = net.CounSelor(
        dataset=args.dataset,
        base=args.base,
        atom_embedding=atom_embedding,
        antecedent_len=args.antecedent_len,
        head=1,
        num_atoms=ap.num_atoms(),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_classes=len(class_names),
        num_data=len(train_dataset),
        tf_model=None,
        avoid_dummy=args.avoid_dummy,
        fix_backbone=args.fix_backbone,
    )
    print('Model Created')
    
    base_path = f'./{args.save_dir}/base_models/'
    base_path += f'base_{args.base}_dataset_{args.dataset}'
    if args.noise_ratio > 0:
        base_path += f'_noise_ratio_{args.noise_ratio}'
    base_path += f'_seed_{args.seed}/'
    tf_base_path = base_path + 'model_best_tf'
    base_path += 'model_best.pt'
    
    if args.base in ['llama2']:
        from peft import PeftModel
        from transformers import LlamaModel
        
        tf_model = None
        model.tf_model = None
        
        tf_model = LlamaModel.from_pretrained("/home/data/llama2/Llama-2-7b-hf", device_map='auto')
        tf_model = PeftModel.from_pretrained(tf_model, tf_base_path)
        
        model.ag.load_state_dict(torch.load(base_path, map_location='cpu'), strict=False)
        model = model.to(gpu)
        model.ag.tf_model = tf_model
        
    else:
        model.ag.tf_model = tf_model
        model.ag.load_state_dict(torch.load(base_path, map_location='cpu'), strict=False)
        print('Model Weight Loaded')

        del tf_model
        model = model.to(gpu)

    print('Model Loaded on GPU')
    loss_func = te.LossFunc(
        model.model_name,
        true_matrix=true_matrix.to(gpu),
        train_y=torch.tensor(ap.train_y).to(gpu),
        num_classes=len(class_names),
        num_data=len(train_dataset),
        reg_lambda=args.reg_lambda,
    )

    dir_prefix = f'{RUN}_{args.base}_dataset_{args.dataset}'
    dir_prefix += f'_antecedent_len_{args.antecedent_len}'
    if dtype == 'tab' and args.num2cat:
        dir_prefix += f'_num2cat'
    if dtype == 'nlp':
        dir_prefix += f'_num_atoms_{args.num_atoms}'
    if args.avoid_dummy:
        dir_prefix += '_avoid_dummy'
    if args.fix_backbone:
        dir_prefix += '_fix_backbone'
    if args.reg_lambda < 1.0:
        dir_prefix += f'_reg_lambda_{args.reg_lambda}'
    if args.noise_ratio > 0:
        dir_prefix += f'_noise_ratio_{args.noise_ratio}'
        
    dir_prefix += f'_seed_{args.seed}'

    if args.only_eval:
        result_path = Path(f'./{args.result_dir}/{RUN}')
        targets = [d for d in result_path.iterdir() if d.name.startswith(dir_prefix)]
        dir_path = Path(f'{targets[-1]}')
        print(f'Directory Path: {str(dir_path)}')
    else:
        now = datetime.now()
        cur_time = now.strftime("%y%m%d:%H:%M:%S")

        dir_path = Path(f'./{args.result_dir}/{RUN}/{dir_prefix}_{cur_time}')
        print(f'Directory Path: {str(dir_path)}')
        dir_path.mkdir(parents=True, exist_ok=True)
        arg_path = dir_path / 'args'
        arg_path.write_text(f'{args}\n', encoding='utf-8')

        model = te.train(
            model=model,
            loss_func=loss_func,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            epochs=args.epochs,
            gpu=gpu,
            class_names=class_names,
            dir_path=dir_path
        )

    best_model_path = dir_path / 'model_best.pt'
    
    if args.base == 'llama2':
        from peft import PeftModel
        from transformers import LlamaModel
        
        tf_model = None
        model.ag.tf_model = None
        
        tf_best_model_path = dir_path / 'model_best_tf'
        tf_model = LlamaModel.from_pretrained("/home/data/llama2/Llama-2-7b-hf", device_map='auto')
        tf_model = PeftModel.from_pretrained(tf_model, tf_best_model_path)
        
        model.load_state_dict(torch.load(str(best_model_path), map_location='cpu'))
        model = model.to(gpu)
        model.ag.tf_model = tf_model
    else:
        model.load_state_dict(torch.load(str(best_model_path), map_location='cpu'))
        model = model.to(gpu)

    te.eval_model(
        model=model,
        loss_func=loss_func,
        test_dataloader=test_dataloader,
        true_matrix=true_matrix.to(gpu),
        gpu=gpu,
        class_names=class_names,
        dir_path=dir_path
    )

    exp_list, result_list = te.get_all_explanation(
        model,
        args.dataset,
        test_data,
        atom_pool=ap,
        true_matrix=true_matrix.to(gpu),
        tf_tokenizer=tf_tokenizer,
        atom_tokenizer=atom_tokenizer,
        gpu=gpu,
        class_names=class_names,
    )

    exp_path = dir_path / 'model_explanation.json'

    with exp_path.open("w", encoding='utf-8') as f:
        json.dump(result_list, f)
