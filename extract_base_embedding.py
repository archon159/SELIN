"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The script to extract embeddings from base models
"""
import shutil
from distutils import dir_util
from pathlib import Path
import torch

# Import from custom files
from selin_utils import dataset as ds
from selin_utils import net
from selin_utils import train_eval as te
from selin_utils import utils

def update_base_model(
    dataset: str,
    base: str,
    seed: int,
    noise_ratio: float,
    result_dir: str,
    save_dir: str,
) -> str:
    """
    The function to update the latest base model
    """
    result_path = Path(f'./{result_dir}/base')

    prefix = f'base_{base}_dataset_{dataset}'
    if noise_ratio > 0:
        prefix += f'_noise_ratio_{noise_ratio}'
    prefix += f'_seed_{seed}'
    cands = sorted([c for c in result_path.iterdir() if c.name.startswith(prefix)])
    target = cands[-1]

    base_update_path = Path(f'./{save_dir}/base_models/{prefix}')
    base_update_path.mkdir(parents=True, exist_ok=True)

    best_base_model_path = target / 'model_best.pt'
    eval_path = target / 'model_eval'

    shutil.copy(str(best_base_model_path), str(base_update_path / 'model_best.pt'))
    shutil.copy(str(eval_path), str(base_update_path / 'model_eval'))
    
    if base in ['llama2']:
        tf_best_base_model_path = target / 'model_best_tf'
        dir_util.copy_tree(str(tf_best_base_model_path), str(base_update_path / 'model_best_tf'))

    return base_update_path

if __name__ == "__main__":
    args = utils.parse_arguments()

    dtype = ds.get_dataset_type(args.dataset)
    btype = ds.get_base_type(args.base)

    assert dtype == btype

    seed = args.seed
    gpu = torch.device(f'cuda:{args.gpu}')
#     gpu = torch.device('cpu')
    utils.reset_seed(seed)

    tf_tokenizer, tf_model, config = net.get_tf_model(args.base)

    # Create datasets
    train_df, valid_df, test_df = ds.load_data(dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset = [
        ds.create_dataset(
            df,
            dataset=args.dataset,
            atom_pool=None,
            atom_tokenizer=None,
            tf_tokenizer=tf_tokenizer,
            config=config
        ) for df in [train_df, valid_df, test_df]]

    train_dataloader, valid_dataloader, test_dataloader = [
        ds.create_dataloader(
            dtset,
            args.batch_size,
            args.num_workers,
            shuffle=False
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
        )
        model = model.to(gpu)

    print('Update the base model to the latest one.')
    base_path = update_base_model(
        args.dataset,
        args.base,
        args.seed,
        args.noise_ratio,
        args.result_dir,
        args.save_dir,
    )

    # We use pre-trained model for NLP bases.
    best_model_path = base_path / 'model_best.pt'
    if btype == 'tab':
        model.load_state_dict(torch.load(best_model_path.resolve(), map_location=gpu), strict=True)
        model = model.to(gpu)

    train_embeddings, valid_embeddings, test_embeddings = [
        te.get_base_embedding(
            model=model,
            train_dataloader=loader,
            gpu=gpu,
        ) for loader in [train_dataloader, valid_dataloader, test_dataloader]]
    
    print(f'Train Embedding Shape: {train_embeddings.shape}')
    embedding_path = base_path / 'train_embeddings.pt'
    print(embedding_path)
    torch.save(train_embeddings, str(embedding_path))

    print(f'Valid Embedding Shape: {valid_embeddings.shape}')
    embedding_path = base_path / 'valid_embeddings.pt'
    print(embedding_path)
    torch.save(valid_embeddings, str(embedding_path))
    
    print(f'Test Embedding Shape: {test_embeddings.shape}')
    embedding_path = base_path / 'test_embeddings.pt'
    print(embedding_path)
    torch.save(test_embeddings, str(embedding_path))