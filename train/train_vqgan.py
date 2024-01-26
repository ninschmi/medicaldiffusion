"Adapted from https://github.com/SongweiGe/TATS"

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from ddpm.diffusion import default
from vq_gan_3d.model import VQGAN3D
from vq_gan_2d.model import VQGAN2D
from train.callbacks import ImageLogger, VideoLogger
from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import WarningCache


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

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

    if cfg.dataset.name == 'FIVES':
        model = VQGAN2D(cfg)
    else:
        model = VQGAN3D(cfg)
    
    # when resume training is true, either resume from the specified checkpoint file
    # or load the most recent checkpoint file in cfg.model.default_root_dir
    dirpath = None 
    version = ''
    #if cfg.model.resume:
    #    if ckpt is not None:
    #        ckpt_folder = ckpt.replace('latest_checkpoint.ckpt', '')
    #        dirpath = ckpt_folder.replace('checkpoints/', '')
    #        version = dirpath.split("/")[-2]
    #        for file in os.listdir(ckpt_folder):
    #            if file.endswith('.ckpt'):
    #                os.rename(os.path.join(ckpt_folder, file),
    #                          os.path.join(ckpt_folder, 'prev_' + file))
    #        ckpt = ckpt.replace('latest_checkpoint.ckpt', 'prev_latest_checkpoint.ckpt')
    #        print('will start from the defined ckpt %s' %
    #            cfg.model.resume_from_checkpoint)
    #    else:
    #        # load the most recent checkpoint file
    #        base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    #        if os.path.exists(base_dir):
    #            log_folder = ckpt_file = ''
    #            version_id_used = 0
    #            for folder in os.listdir(base_dir):
    #                if os.path.exists(os.path.join(base_dir, folder,'checkpoints/latest_checkpoint.ckpt')):
    #                    version_id = int(folder.split('_')[1])
    #                    if version_id > version_id_used:
    #                        version_id_used = version_id
    #                        log_folder = folder
    #            if len(log_folder) > 0:
    #                ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
    #                for fn in os.listdir(ckpt_folder):
    #                    if fn == 'latest_checkpoint.ckpt':
    #                        ckpt_file = 'latest_checkpoint_prev.ckpt'
    #                        os.rename(os.path.join(ckpt_folder, fn),
    #                                  os.path.join(ckpt_folder, ckpt_file))
    #                        ckpt = os.path.join(
    #                        ckpt_folder, ckpt_file)
    #                        print('will start from the most recent ckpt %s' %
    #                            cfg.model.resume_from_checkpoint)
    #                        dirpath = os.path.join(base_dir, log_folder)
    #                        version = dirpath.split("/")[-1]
    #                    if fn == 'latest_checkpoint-v1.ckpt' or fn == 'latest_checkpoint-v2.ckpt':
    #                        os.remove(os.path.join(ckpt_folder, fn))
    #            else:
    #                print('no checkpoints found in %s, will start training from scratch' % base_dir)
    #        else:
    #            print('%s does not exist, will start training from scratch' % base_dir)
    #else:
    #    ckpt = None
    ckpt = None

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=1500, max_videos=4, clamp=True))

    accelerator = 'auto'
    strategy = 'auto'
    devices = 'auto'
    if cfg.model.gpus > 0:
        accelerator = 'gpu'
        devices=cfg.model.gpus
        if cfg.model.gpus > 1:
            # Explicitly specify the process group backend if you choose to
            ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)
            strategy = ddp
            #strategy = 'ddp'

    # when resuming training, make sure that log_dir is set accordingly (keep saving the logs in the same directory)
    logger = None
    #if dirpath is not None:
    #    try:
    #        logger = TensorBoardLogger(save_dir=cfg.model.default_root_dir, version=int(version.split('_')[-1]))
    #    except:
    #        warning_cache = WarningCache()
    #        warning_cache.warn(
    #            "Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch`"
    #            " package, due to potential conflicts with other packages in the ML ecosystem. For this reason,"
    #            " `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard`"
    #            " or `tensorboardX` packages are found."
    #            " Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default"
    #        )
    #        logger = CSVLogger(save_dir=cfg.model.default_root_dir, version=int(version.split('_')[-1]))  # type: ignore[assignment]
    #        #load metrics from metrics.csv file
    #        metrics_file = logger.log_dir + '/' + logger.experiment.NAME_METRICS_FILE
    #        with open(metrics_file) as file:
    #            keys = file.readline().strip().split(",")
    #        logger.experiment.metrics_keys=keys

    trainer = pl.Trainer(
        devices=devices,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        accelerator=accelerator,
        strategy=strategy,
        logger=logger
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == '__main__':
    run()
