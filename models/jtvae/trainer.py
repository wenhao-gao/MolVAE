import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from tqdm.auto import tqdm

from models.trainer import Trainer
from models.jtvae.mol_tree import Vocab, MolTree


class JTVAETrainer(Trainer):
    def __init__(self, config):
        self.config = config

    def get_vocabulary(self, data):
        vocab = [x.strip("\r\n ") for x in open(config.vocab)]
        return Vocab(vocab)

    def _train_epoch(self, model, epoch, tqdm_data, optimizer=None):

        if optimizer is None:
            model.eval()
        else:
            model.train()

        for batch in tqdm_data:
            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc = model(batch, beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
                optimizer.step()
            except Exception as e:
                print(e)
                continue

        postfix = {
            'epoch': epoch,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'

        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):
        device = model.device

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, configs.anneal_rate)

        param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None])

        model.zero_grad()
        for epoch in range(config.epoch):
            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, epoch,
                                        tqdm_data, kl_weight, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(),
                           self.config.model_save[:-3] +
                           '_{0:03d}.pt'.format(epoch))
                model = model.to(device)

            # Epoch end
            lr_annealer.step()

    def fit(self, model, train_data, val_data=None):
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        if args.load_epoch > 0:
            model.load_state_dict(torch.load(configs.save_dir + "/model.iter-" + str(configs.load_epoch)))

        print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

        loader = MolTreeFolder(args.train, vocab, args.batch_size, num_workers=4)


