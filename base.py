"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The script to train base models
"""
from pathlib import Path
from datetime import datetime
import torch
from torch import nn

# Import from custom files
from selin_utils import dataset as ds
from selin_utils import net
from selin_utils import train_eval as te
from selin_utils import utils

RUN = 'base'

if __name__ == "__main__":
    args = utils.parse_arguments()

    dtype = ds.get_dataset_type(args.dataset)
    btype = ds.get_base_type(args.base)
    assert dtype == btype

    seed = args.seed
    if args.base in ['llama2']:
        gpu = torch.device(f'cuda')
    else:
        gpu = torch.device(f'cuda:{args.gpu}')
#     gpu = torch.device(f'cpu')
    utils.reset_seed(seed)

    tf_tokenizer, tf_model, config = net.get_tf_model(args.base)
    # Create datasets
    
    train_data, valid_data, test_data = ds.load_data(dataset=args.dataset)
    
    train_dataset = ds.create_dataset(
        train_data,
        is_train=True,
        dataset=args.dataset,
        atom_pool=None,
        atom_tokenizer=None,
        tf_tokenizer=tf_tokenizer,
        config=config,
        noise_ratio=args.noise_ratio,
    )
    
    valid_dataset, test_dataset = [
        ds.create_dataset(
            data,
            is_train=False,
            dataset=args.dataset,
            atom_pool=None,
            atom_tokenizer=None,
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
    
    if dtype=='nlp':
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
        embedding_dim = 768
    elif dtype=='tab':
        input_dim = train_dataset.x.shape[1]
        if args.base == 'dnn':
            hidden_dim = args.hidden_dim
            embedding_dim = 512
        elif args.base == 'fttransformer':
            hidden_dim = args.hidden_dim
            embedding_dim = 512
    else:
        raise ValueError(f'Dataset type {dtype} is not supported.')

    # Load class names
    class_names = ds.get_class_names(args.dataset)

    if args.base in ['llama2']:
        model = net.BaseModel(
            dataset=args.dataset,
            base=args.base,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            tf_model=None,
            num_classes=len(class_names),
            fix_backbone=args.fix_backbone
        )
        model = model.to(gpu)
        model.tf_model = tf_model
    else:
        model = net.BaseModel(
            dataset=args.dataset,
            base=args.base,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            tf_model=tf_model,
            num_classes=len(class_names),
            fix_backbone=args.fix_backbone
        )
        model = model.to(gpu)
        
    loss_func = te.LossFunc(model.model_name)
    dir_prefix = f'{RUN}_{args.base}_dataset_{args.dataset}'
    if args.fix_backbone:
        dir_prefix += '_fix_backbone'
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
        model.tf_model = None
        
        tf_best_model_path = dir_path / 'model_best_tf'
        tf_model = LlamaModel.from_pretrained("/home/data/llama2/Llama-2-7b-hf", device_map='auto')
        tf_model = PeftModel.from_pretrained(tf_model, tf_best_model_path)
        
        model.load_state_dict(torch.load(str(best_model_path), map_location='cpu'))
        model = model.to(gpu)
        model.tf_model = tf_model
    else:
        model.load_state_dict(torch.load(str(best_model_path), map_location='cpu'))
        model = model.to(gpu)

    te.eval_model(
        model=model,
        loss_func=loss_func,
        test_dataloader=test_dataloader,
        true_matrix=None,
        gpu=gpu,
        class_names=class_names,
        dir_path=dir_path
    )
