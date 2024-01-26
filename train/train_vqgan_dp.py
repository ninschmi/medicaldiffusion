"Adapted from https://github.com/SongweiGe/TATS"

import numpy as np
import os
import random
import pickle
import torch
#TODO from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from ddpm.diffusion import default
from vq_gan_3d.model import VQGAN3D
from vq_gan_2d.model.vqgan_torch import VQGAN2D_torch
from train.callbacks_torch import ImageLogger, VideoLogger
from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict
from train.logger import TensorBoardLogger
from torch.nn.parallel import DistributedDataParallel as DDP

def seed_everything(seed: int):
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    Args:
        seed: the integer value seed for global random state
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    if not (min_seed_value <= seed <= max_seed_value):
        print(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = random.randint(min_seed_value, max_seed_value)  # noqa: S311
    print(f"Seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def done(max_steps, global_step, max_epochs, epoch) -> bool:
    # TODO: Move track steps inside training loop and move part of these condition inside training loop
    stop_steps =  max_steps != -1 and global_step >=  max_steps
    if stop_steps:
        print(f"Training stopped: `max_steps={max_steps!r}` reached.")
        return True
    assert isinstance(max_epochs, int)
    stop_epochs = max_epochs != -1 and epoch >= max_epochs
    if stop_epochs:
        print(f"Training stopped: `max_epochs={max_epochs!r}` reached.")
        return True
    return False

def is_picklable(obj: object) -> bool:
    """Tests if an object can be pickled."""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, AttributeError, RuntimeError, TypeError):
        return False
    
def main_params(optimizer):
    for group in optimizer.param_groups:
        yield from group["params"]


def train_model(cfg, model, train_dataloader, val_dataloader, image_logger, device) -> None:
    current_epoch = 0
    global_step = 0

    # Training loop
    print("Starting Training Loop...")
    while not done(cfg.model.max_steps, global_step, cfg.model.max_epochs, current_epoch): #replaces: for epoch in range(cfg.model.max_epochs) to allow infinite training_loop
        print(current_epoch)
        
        # Training
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device=device) for k, v in batch.items()}
            model.train()   # make sure perceptual model remains in eval
            model.training_step(batch, current_epoch, global_step)

            #if global_step % 3000 == 0:
            #    torch.save()
            #    filename='{epoch}-{step}-{train/recon_loss:.2#f}'
#
            #    checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
            #    self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
            #    self.strategy.barrier("Trainer.save_checkpoint")

            # ON TRAIN BATCH END - on_train_batch_end
            image_logger.log_img(batch, batch_idx, global_step, current_epoch, split="train")

            global_step += 1
            #3D
            #video_logger.log_vid(batch, batch_idx, global_step, current_epoch, split="train")

        # Validation
        model.eval()
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = {k: v.to(device=device, non_blocking=True) for k, v in val_batch.items()}
                model.eval()
                model.validation_step(val_batch)

                # ON VALIDATION BATCH END - on_validation_batch_end
                image_logger.log_img(batch, batch_idx, global_step, current_epoch, split="val")  
                #3D
                #video_logger.log_vid(batch, batch_idx, global_step, current_epoch, split="val")      

        

        current_epoch += 1

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches
    ckpt = cfg.model.resume_from_checkpoint

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        cfg.model.lr, accumulate, ngpu/8, bs/4, base_lr))

    # Logger
    logger = TensorBoardLogger(root_dir=cfg.model.default_root_dir)

    # Model
    if cfg.dataset.name == 'FIVES':
        model = VQGAN2D_torch(cfg, logger)
    else:
        model = VQGAN3D(cfg)
    nn_module = model

    image_logger = ImageLogger(
        batch_frequency=750, model=model, save_dir=logger.log_dir, max_images=4, clamp=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu")
    num_processes = torch.cuda.device_count() if ngpu > torch.cuda.device_count() else ngpu
    if num_processes > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(module=model)
        nn_module = model.module
    
    model.to(device)


    try:    
        # clean hparams
        if hasattr(model, "hparams"):
            """Removes all unpicklable entries from hparams."""
            del_attrs = [k for k, v in model.hparams.items() if not is_picklable(v)]
            for k in del_attrs:
                print(f"attribute '{k}' removed from hparams because it cannot be pickled")
                del model.hparams[k]
        # HOOKS
        #TODO
        # on_fit_start (callback_hooks and lightning_module_hook)

        
        train_model(cfg, nn_module, train_dataloader, val_dataloader, image_logger, device)

        #teardown and post-training clean up
        print(f"tearing down strategy")
        if isinstance(model, torch.nn.DataParallel):
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]

        optimizers = model.module.optimizers if isinstance(model, torch.nn.DataParallel) else model.optimizers
        for opt in optimizers:
            for p, v in opt.state.items():
                kwargs = {}
                # Don't issue non-blocking transfers to CPU
                # Same with MPS due to a race condition bug: https://github.com/pytorch/pytorch/issues/83015
                if isinstance(v, torch.Tensor) and isinstance(torch.device("cpu"), torch.device) and torch.device("cpu").type not in ("cpu", "mps"):
                    kwargs["non_blocking"] = True
                data_output = v.to(torch.device("cpu"), **kwargs)
                opt.state[p] = data_output if data_output is not None else v

        if model is not None:
            print(f"moving model to CPU")
            model.cpu()

        # HOOKS
        #TODO
        # on_fit_end (callback_hooks and lightning_module_hook)
            
        # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
        # It might be related to xla tensors blocked when moving the cpu kill loggers.
        logger.finalize("success")

        train_status = 2

    ## TODO: Unify both exceptions below, where `KeyboardError` doesn't re-raise
    except KeyboardInterrupt as exception:
        print("Detected KeyboardInterrupt, attempting graceful shutdown...")
        # user could press Ctrl+c many times... only shutdown once
        if not train_status == 3: #interrupted
            train_status = 3
    #        _call_callback_hooks(trainer, "on_exception", exception)
    #        trainer.strategy.on_exception(exception)
            logger.finalize("failed")
    except BaseException as exception:
        print("Detected BaseException, attempting graceful shutdown...")
        train_status = 3 #interrupted
    #    _call_callback_hooks(trainer, "on_exception", exception)
    #    trainer.strategy.on_exception(exception)
        logger.finalize("failed")
    #    trainer._teardown()
    #    # teardown might access the stage so we reset it after
    #    trainer.state.stage = None
    #    raise

if __name__ == '__main__':
    run()