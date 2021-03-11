import argparse
import os
import sys
import torch
import rdkit

from utils.script_utils import add_train_args, set_seed
from models.models_storage import ModelsStorage
from tdc.generation import MolGen

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        add_train_args(
            MODELS.get_model_train_parser(model)(
                subparsers.add_parser(model)
            )
        )
    return parser


def main(model, config):
    set_seed(config.seed)
    device = torch.device(config.device)

    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    config.config_save = os.path.join(config.checkpoint_dir, model + config.experimental_stuff + '_config.pt')
    config.model_save = os.path.join(config.checkpoint_dir, model + config.experimental_stuff + '_model.pt')
    config.vocab_save = os.path.join(config.checkpoint_dir, model + config.experimental_stuff + '_vocab.txt')

    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    
    trainer = MODELS.get_model_trainer(model)(config)

    if config.processed_data is not None:
        train_data = trainer.load_train_data()
        val_data = trainer.load_val_data()
    else:
        data = MolGen(name = 'MOSES', path = config.data_path)
        split = data.get_split(method = 'random', seed = config.data_seed, frac = [0.8, 0.0, 0.2])

        train_data = split['train']['smiles'].tolist()
        val_data = split['test']['smiles'].tolist()

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), \
            'vocab_load path does not exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data)

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    model = MODELS.get_model_class(model)(vocab, config).to(device)
    trainer.fit(model, train_data, val_data)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)

