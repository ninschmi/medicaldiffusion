"Adapted from https://github.com/SongweiGe/TATS"

import copy
import numpy as np
import os
import sys
import random
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from ddpm.diffusion import default
from vq_gan_3d.model import VQGAN3D
from vq_gan_2d.model.vqgan_torch import VQGAN2D_torch
from vq_gan_3d.model.vqgan_torch import VQGAN3D_torch
from train.callbacks_torch import ImageLogger, VideoLogger
from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict
from train.logger import TensorBoardLogger
from train.model_checkpoint import ModelCheckpoint
from train.checkpoint import Checkpoint
from contextlib import nullcontext
import socket
from threading import Thread
from typing import Any, Callable, List, Optional, Sequence, Tuple, MutableMapping
import pickle
from train.utils import _ChildProcessObserver, AttributeDict, collect_init_args
import inspect
import types
from dataclasses import fields, is_dataclass
from argparse import Namespace

# train_status
#   0 INITIALIZING = "initializing"
#   1 RUNNING = "running"
#   2 FINISHED = "finished"
#   3 INTERRUPTED = "interrupted"
#   stopped == FINISHED or self.INTERRUPTED

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

def init_process(rank, world_size):
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available. Cannot initialize distributed process group")
    if torch.distributed.is_initialized():
        print("torch.distributed is already initialized. Exiting early")
    else:
        # DDP Environment variables
        # Environment variables which need to be
        # set when using c10d's default "env"
        # initialization mode.
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = '29500' #"43633"
        #for key, value in os.environ.items():
        #    print(f"{key}: {value}")
        print(f"Initializing distributed: GLOBAL_RANK: {rank}, MEMBER: {rank + 1}/{world_size}")
        #TODO: switch backend='nccl'?
        torch_distributed_backend = 'gloo'
        torch.distributed.init_process_group(backend=torch_distributed_backend, world_size=world_size, rank=rank)
        # On rank=0 let everyone know training is starting
        if rank == 0:
            print(
                f"{'-' * 100}\n"
                f"distributed_backend={torch_distributed_backend}\n"
                f"All distributed processes registered. Starting with {world_size} processes\n"
                f"{'-' * 100}\n")
        
def cleanup():
    torch.distributed.destroy_process_group()

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

def train_model(cfg, model, train_dataloader, val_dataloader, logger, image_logger, video_logger, device, callbacks) -> None:
    current_epoch = 0
    global_step = 0 # batches_that_stepped
    num_optimizers = len(model.optimizers)

    #TODO connect checkpoint training resumed from with model_checkpoint.ckpt_path = ckpt

    # Training loop
    print("Starting Training Loop...")
    while not done(cfg.model.max_steps, global_step, cfg.model.max_epochs, current_epoch): #replaces: for epoch in range(cfg.model.max_epochs) to allow infinite training_loop
        print("current epoch: ", current_epoch)
        
        # Training
        logger.update_epoch(current_epoch)
        logger.reset_results()

        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
            model.train()   # make sure perceptual model remains in eval

            # ON BATCH START
            logger.on_train_batch_start(batch)

            model.training_step(batch, current_epoch, global_step)

            # ON TRAIN BATCH END - on_train_batch_end
            image_logger.log_img(batch, batch_idx, (global_step + 1) * num_optimizers, current_epoch, split="train")
            #3D
            if video_logger is not None:
                video_logger.log_vid(batch, batch_idx, (global_step + 1) * num_optimizers, current_epoch, split="train")

            for cb in callbacks:
                cb.on_train_batch_end((global_step+ 1) * num_optimizers, current_epoch)

            # ON BATCH END
            logger.on_train_batch_end(step=global_step)
            
            global_step += 1
        print(f"training step completed after {global_step} steps")
        # Validation
        model.eval()
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                val_batch = {k: v.to(device=device, non_blocking=True) for k, v in val_batch.items()}
                model.eval()

                # ON BATCH START
                logger.on_val_batch_start(val_batch)

                model.validation_step(val_batch)

                # ON VALIDATION BATCH END - on_validation_batch_end
                image_logger.log_img(batch, batch_idx, (global_step + 1) * num_optimizers, current_epoch, split="val")  
                #3D
                if video_logger is not None:
                    video_logger.log_vid(batch, batch_idx, (global_step + 1) * num_optimizers, current_epoch, split="val")

                # ON BATCH END
                logger.on_val_batch_end(step=val_batch_idx+(global_step-1)*len(val_dataloader))

        print(f"validation step completed after {global_step} steps")

        logger.on_epoch_end()

        for cb in callbacks:
            cb.on_validation_end(global_step * num_optimizers, current_epoch)
            cb.on_train_epoch_end(global_step * num_optimizers, current_epoch)
        
        #log metrics after every epoch 
        logger.log_results(step=global_step-1)
        #reset result collection for next epoch
        logger.reset_results()

        current_epoch += 1
    
    # teardown
    logger.teardown()


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    # set seed for reproducibility
    seed_everything(cfg.model.seed)

    # Dataset, Dataloader
    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, shuffle=True)  #sampler=sampler
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    # hyperparameters
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
    has_video_logger = True

    # Model
    if cfg.dataset.name == 'FIVES':
        model = VQGAN2D_torch(cfg, logger)
        has_video_logger = False
    elif cfg.dataset.name == 'MIDAS' or cfg.dataset.name == 'OPENNEURO' or cfg.dataset.name == 'ADNI':
        model = VQGAN3D_torch(cfg, logger)
    else:
        model = VQGAN3D(cfg)

    train_status = 0

    optimizers = model.optimizers if cfg.model.extended else model.optimizers[:-1]

    checkpoint = Checkpoint(model, optimizers)

    # push all model checkpoint callbacks to the end
    # it is important that these are the last callbacks to run
    callbacks = []
    #save top 3 for all validation losses individually
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='val/recon_loss/{val/recon_loss:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/recon_loss_image',
                     save_top_k=3, mode='min', filename='val/recon_loss_image/{val/recon_loss_image:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/recon_loss_mask',
                     save_top_k=3, mode='min', filename='val/recon_loss_mask/{val/recon_loss_mask:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/perceptual_loss',
                     save_top_k=3, mode='min', filename='val/perceptual_loss/{val/perceptual_loss:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/perceptual_loss_image',
                     save_top_k=3, mode='min', filename='val/perceptual_loss_image/{val/perceptual_loss_image:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/perceptual_loss_mask',
                     save_top_k=3, mode='min', filename='val/perceptual_loss_mask/{val/perceptual_loss_mask:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/commitment_loss',
                     save_top_k=3, mode='min', filename='val/commitment_loss/{val/commitment_loss:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/g_image_loss',
                     save_top_k=3, mode='min', filename='val/g_image_loss/{val/g_image_loss:.2f}', logger=logger))
    if has_video_logger:
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/g_video_loss',
                     save_top_k=3, mode='min', filename='val/g_video_loss/{val/g_video_loss:.2f}', logger=logger))
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/g_mask_image_loss',
                         save_top_k=3, mode='min', filename='val/g_mask_image_loss/{val/g_mask_image_loss:.2f}', logger=logger))
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/g_mask_video_loss',
                         save_top_k=3, mode='min', filename='val/g_mask_video_loss/{val/g_mask_video_loss:.2f}', logger=logger))
    else:
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/g_mask_loss',
                         save_top_k=3, mode='min', filename='val/g_mask_loss/{val/g_mask_loss:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/image_gan_feat_loss',
                     save_top_k=3, mode='min', filename='val/image_gan_feat_loss/{val/image_gan_feat_loss:.2f}', logger=logger))
    if has_video_logger:
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/video_gan_feat_loss',
                         save_top_k=3, mode='min', filename='val/video_gan_feat_loss/{val/video_gan_feat_loss:.2f}', logger=logger))
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/mask_image_gan_feat_loss',
                         save_top_k=3, mode='min', filename='val/mask_image_gan_feat_los/{val/mask_image_gan_feat_los:.2f}', logger=logger))
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/mask_video_gan_feat_loss',
                         save_top_k=3, mode='min', filename='val/mask_video_gan_feat_loss/{val/mask_video_gan_feat_loss:.2f}', logger=logger))
    else:
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/mask_gan_feat_loss',
                         save_top_k=3, mode='min', filename='val/mask_gan_feat_los/{val/mask_gan_feat_los:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/aeloss',
                     save_top_k=3, mode='min', filename='val/aeloss/{val/aeloss:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/aeloss_image',
                     save_top_k=3, mode='min', filename='val/aeloss_image/{val/aeloss_image:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/aeloss_mask',
                     save_top_k=3, mode='min', filename='val/aeloss_mask/{val/aeloss_mask:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/discloss',
                     save_top_k=3, mode='min', filename='val/discloss/{val/discloss:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/discloss_image',
                     save_top_k=3, mode='min', filename='val/discloss_image/{val/discloss_images:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/discloss_mask',
                     save_top_k=3, mode='min', filename='val/discloss_mask/{val/discloss_mask:.2f}', logger=logger))
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/d_image_loss',
                     save_top_k=3, mode='min', filename='val/d_image_loss/{val/d_image_loss:.2f}', logger=logger))
    if has_video_logger:
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/d_video_loss',
                         save_top_k=3, mode='min', filename='val/d_video_loss/{val/d_video_loss:.2f}', logger=logger))
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/d_mask_image_loss',
                         save_top_k=3, mode='min', filename='val/d_mask_image_loss/{val/d_mask_image_loss:.2f}', logger=logger))
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/d_mask_video_loss',
                     save_top_k=3, mode='min', filename='val/d_mask_video_loss/{val/d_mask_video_loss:.2f}', logger=logger))
    else:
        callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/d_mask_loss',
                         save_top_k=3, mode='min', filename='val/d_mask_loss/{val/d_mask_loss:.2f}', logger=logger))
    #save at every 100 epochs
    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, every_n_epochs=100, save_top_k=-1, save_last=True,
                     filename='train/{epoch}-{step}', logger=logger))
    #callbacks.append(ModelCheckpoint(checkpoint=checkpoint, monitor='val/recon_loss',
    #                 save_top_k=3, mode='min', filename='latest_checkpoint', logger=logger))
    #callbacks.append(ModelCheckpoint(checkpoint=checkpoint, every_n_train_steps=3000,
    #                 save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}', logger=logger))
    #callbacks.append(ModelCheckpoint(checkpoint=checkpoint, every_n_train_steps=10000, save_top_k=-1,
    #                 filename='{epoch}-{step}-10000-{train/recon_loss:.2f}', logger=logger))
    image_logger = ImageLogger(
        batch_frequency=750, model=model, save_dir=logger.log_dir, max_images=4, clamp=True)
    if has_video_logger:
        video_logger = VideoLogger(
            batch_frequency=1500, model=model, save_dir=logger.log_dir, max_videos=4, clamp=True)
    else:
        video_logger = None
    
    #os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # device
    # GPU
    if torch.cuda.is_available() and ngpu > 0:
        num_processes = torch.cuda.device_count() if ngpu > torch.cuda.device_count() else ngpu
        world_size = num_processes

        device = num_processes
        parallel_device = [torch.device("cuda", i) for i in range(device)]
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        root_device = parallel_device[local_rank]
        print(f"{model}: moving model to device [{root_device}]...")

        # TODO refactor input from trainer to local_rank @four4fish
        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        print(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")
        # strangely, the attribute function be undefined when torch.compile is used
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            # https://github.com/pytorch/pytorch/issues/95668
            torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()

        if num_processes > 1:
            # use DistributedDataParallel (DDP) if more than one GPU is available
            import subprocess
            # return trainer.strategy.launcher.launch(train_model, cfg, model, train_dataloader, val_dataloader, image_logger, device)
        
            # bookkeeping of spawned processes
            procs = []  # reset in case it's called twice; List[subprocess.Popen]

            # allow the user to pass the node rank
            os.environ["NODE_RANK"] = '0'
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            #os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
            #os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())
            #os.environ["WORLD_SIZE"] = f"{self.num_processes * self.num_nodes}"

            for loc_rank in range(1, num_processes):
                env_copy = os.environ.copy()
                env_copy["LOCAL_RANK"] = f"{loc_rank}"

                hydra_in_use = hydra.core.hydra_config.HydraConfig.initialized()

                if hydra_in_use:
                    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
                    from hydra.utils import get_original_cwd, to_absolute_path

                    # when user is using hydra find the absolute path
                    if __main__.__spec__ is None:  # pragma: no-cover
                        command = [sys.executable, to_absolute_path(sys.argv[0])]
                    else:
                        command = [sys.executable, "-m", __main__.__spec__.name]

                    command += sys.argv[1:]

                    cwd = get_original_cwd()
                    rundir = f'"{hydra.core.hydra_config.HydraConfig.get().run.dir}"'
                    # Set output_subdir null since we don't want different subprocesses trying to write to config.yaml
                    command += [f"hydra.run.dir={rundir}", f"hydra.job.name=train_ddp_process_{loc_rank}", "hydra.output_subdir=null"]
                else:
                    cwd = None

                    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

                    if __main__.__spec__ is None:  # pragma: no-cover
                        command = [sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:]
                    else:
                        command = [sys.executable, "-m", __main__.__spec__.name] + sys.argv[1:]

                new_process = subprocess.Popen(command, env=env_copy, cwd=cwd)
                procs.append(new_process)

                monitor_thread = Thread(
                    target=_ChildProcessObserver(child_processes=procs, main_pid=os.getpid()),
                    daemon=True,  # thread stops if the main process exits
                )
                monitor_thread.start()

                if "OMP_NUM_THREADS" not in os.environ:
                    if num_processes < 1:
                        raise ValueError(f"`num_processes` should be >= 1, got {num_processes}.")
                    if hasattr(os, "sched_getaffinity"):
                        num_cpus_available = len(os.sched_getaffinity(0))
                    cpu_count = os.cpu_count()
                    if cpu_count is None:
                        num_cpus_available = 1
                    else:
                        num_cpus_available = cpu_count
                    num_threads = max(1, num_cpus_available // num_processes)
                    torch.set_num_threads(num_threads)
                    os.environ["OMP_NUM_THREADS"] = str(num_threads)

            node_rank = int(os.environ.get("NODE_RANK", 0))
            global_rank = node_rank * num_processes + local_rank
            #rank_zero_only.rank = global_rank

            init_process(global_rank, world_size)

            if root_device.type != "cuda":
                raise ValueError(f"Device should be GPU, got {root_device} instead")
            if torch.cuda.get_device_capability(root_device)[0] >= 8:  # Ampere and later leverage tensor cores, where this setting becomes useful
                # check that the user hasn't changed the precision already, this works for both `allow_tf32 = True` and
                # `set_float32_matmul_precision`
                if torch.get_float32_matmul_precision() == "highest":  # default
                    print(
                        f"You are using a CUDA device ({torch.cuda.get_device_name(root_device)!r}) that has Tensor Cores. To properly"
                        " utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off"
                        " precision for performance. For more details, read https://pytorch.org/docs/stable/generated/"
                        "torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"
                    )
                # note: no need change `torch.backends.cudnn.allow_tf32` as it's enabled by default:
                # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            torch.cuda.set_device(root_device)


            # strategy will configure model and move it to the device

            # TODO refactor input from trainer to local_rank @four4fish
            # set the correct cuda visible devices (using pci order)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            print(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")
            # strangely, the attribute function be undefined when torch.compile is used
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                # https://github.com/pytorch/pytorch/issues/95668
                torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()

            # move the model to the correct device
            print(f"moving model to device [{root_device}]...")
            assert model is not None
            model.to(root_device)

            print(f"configuring DistributedDataParallel") 

            device_ids = None if root_device.type == "cpu" else [root_device.index]
            print(f"setting up DDP model with device ids: {device_ids}")
            # https://pytorch.org/docs/stable/notes/cuda.html#id5
            ctx = torch.cuda.stream(torch.cuda.Stream()) if device_ids is not None else nullcontext()
            with ctx:
                model = DDP(module=model, device_ids=device_ids)

            # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
            # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
            if root_device.type == "cuda":
                assert isinstance(model, DDP)
            #    _register_ddp_comm_hook(model=self.model, ddp_comm_state=self._ddp_comm_state,
            #        ddp_comm_hook=self._ddp_comm_hook, ddp_comm_wrapper=self._ddp_comm_wrapper)

            # if not in FITTING
            # we need to manually synchronize the module's states since we aren't using the DDP wrapper
            #_sync_module_states(self.model)

        else:
            # Use a single GPU
            if cfg.model.device:
                device = torch.device(cfg.model.device)
            else:
                device = torch.device("cuda:0")
            model = model.to(device)
            root_device = device
    # CPU
    else:
        # use cpu if CUDA is not available
        device = torch.device("cpu")
        model = model.to(device)
        root_device = device

    optimizers = model.module.optimizers if isinstance(model, DDP) else model.optimizers
    optimizers = optimizers if cfg.model.extended else model.optimizers[:-1]
    # set up optimizers after the wrapped module has been moved to the device
    for opt in optimizers:
        for p, v in opt.state.items():
            kwargs = {}
            if isinstance(v, torch.Tensor) and isinstance(root_device, torch.device) and root_device.type not in ("cpu", "mps"):
                kwargs["non_blocking"] = True
            data_output = v.to(root_device, **kwargs)
            opt.state[p] = data_output if data_output is not None else v

   
    # Explicitly specify the process group backend if you choose to
    #ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)
    #on_train_batch_start()
    # accumulate_grad_batches=cfg.model.accumulate_grad_batches

    try:    
        #self.state.fn = TrainerFn.FITTING
        #self.state.status = TrainerStatus.RUNNING
        #self.training = True 

        #assert self.state.fn is not None
        ckpt = cfg.model.resume_from_checkpoint #parse ckpt currently not supported

        # clean hparams
        if hasattr(model, "hparams"):
            """Removes all unpicklable entries from hparams."""
            del_attrs = [k for k, v in model.hparams.items() if not is_picklable(v)]
            for k in del_attrs:
                print(f"attribute '{k}' removed from hparams because it cannot be pickled")
                del model.hparams[k]

        if ckpt:
            ## check if we should delay restoring checkpoint till later, currently set to False (depends on strategy deployed)
            #if not self.strategy.restore_checkpoint_after_setup:
            print(f"restoring module and callbacks from checkpoint path: {ckpt}")
            #restore_modules_and_callbacks
            checkpoint._restore_modules_and_callbacks(ckpt)


        # save hyperparameters
        current_frame = inspect.currentframe()
        if current_frame:
            frame = current_frame.f_back
    
        if not isinstance(frame, types.FrameType):
            raise AttributeError("There is no `frame` available while being required.")

        init_args = {'cfg': cfg}
        #for local_args in collect_init_args(frame, [], classes=(VQGAN3D_torch,)):
        #    init_args.update(local_args)
       
        ignore = []
        ignore = list(set(ignore))
        init_args = {k: v for k, v in init_args.items() if k not in ignore}

        # take all arguments
        hp = init_args
        hparams_name = "kwargs" if hp else None

        # `hparams` are expected here
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, (bool, int, float, str)):
            raise ValueError(f"Primitives {(bool, int, float, str)} are not allowed.")
        elif not isinstance(hp, (AttributeDict, MutableMapping, Namespace)):
            raise ValueError(f"Unsupported config type of {type(hp)}.")
    
        hparams = AttributeDict()
        if isinstance(hp, dict) and isinstance(hparams, dict):
            hparams.update(hp)
        else:
            hparams = hp

        for k, v in hparams.items():
            if isinstance(v, nn.Module):
                #rank_zero_warn()
                print(
                    f"Attribute {k!r} is an instance of `nn.Module` and is already saved during checkpointing."
                    f" It is recommended to ignore them using `self.save_hyperparameters(ignore=[{k!r}])`."
                )

        # make a deep copy so there are no other runtime changes reflected
        hparams_initial = copy.deepcopy(hparams)

        # log hyperparameters
        if hparams_initial is not None:
            logger.log_hyperparams(hparams_initial)
        logger.save()

        if ckpt: 
            # restore optimizers, etc.
            print(f"restoring training state")
            checkpoint.restore_training_state()
            checkpoint.resume_end()

         # wait for all to join if on distributed
        if num_processes > 1:
            assert torch.distributed.is_available() and torch.distributed.is_initialized()

            if torch.distributed.get_backend() == "nccl":
                torch.distributed.barrier(device_ids=[root_device.index] or None)   # None if root_device.type == "cpu"
            else:
                torch.distributed.barrier()

        # RUN THE TRAINING
        train_model(cfg, model, train_dataloader, val_dataloader, logger, image_logger, video_logger, device, callbacks)

        #teardown and post-training clean up
        print(f"tearing down strategy")
        if isinstance(model, DDP):
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]

        optimizers = model.module.optimizers if isinstance(model, DDP) else model.optimizers
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

        if root_device.type == "cuda":
            # strangely, the attribute function be undefined when torch.compile is used
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                # https://github.com/pytorch/pytorch/issues/95668
                torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()

            
        # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
        # It might be related to xla tensors blocked when moving the cpu kill loggers.
        logger.finalize("success")

        train_status = 2

        #assert self.state.stopped
        #self.training = False

    ## TODO: Unify both exceptions below, where `KeyboardError` doesn't re-raise
    except KeyboardInterrupt as exception:
        print("Detected KeyboardInterrupt, attempting graceful shutdown...")
        # user could press Ctrl+c many times... only shutdown once
        if not train_status == 3: #interrupted
            train_status = 3
    #        _call_callback_hooks(trainer, "on_exception", exception)
    #        trainer.strategy.on_exception(exception)
            logger.finalize("failed")
    #except BaseException as exception:
    #    train_status = 3 #interrupted
    ##    _call_callback_hooks(trainer, "on_exception", exception)
    ##    trainer.strategy.on_exception(exception)
    #    logger.finalize("failed")
    ##    trainer._teardown()
    ##    # teardown might access the stage so we reset it after
    ##    trainer.state.stage = None
    ##    raise




    #def _run(self, model: LightningModule, ckpt_path: Optional[str] = None) -> None:
        #self._callback_connector._attach_model_callbacks()
        #self._callback_connector._attach_model_logging_functions()
        #_verify_loop_configurations(self)
        # hook; preparing data
        #self._data_connector.prepare_data()
        #self.__setup_profiler()
        #call._call_setup_hook(self)  # allow user to setup lightning_module in accelerator environment

        #barrier in setup???
        #assert torch.distributed.is_available() and torch.distributed.is_initialized()
#
        #if torch.distributed.get_backend() == "nccl":
        #    torch.distributed.barrier(device_ids=[root_device.index] or None)   # None if root_device.type == "cpu"
        #else:
        #    torch.distributed.barrier()

        #self._signal_connector.register_signal_handlers()

        # RUN THE TRAINING

        #self.checkpoint_io.teardown()



        #args = (Tensor, move_data_to_device, "cpu")
        #self._logged_metrics = apply_to_collection(self._logged_metrics, *args)
        #self._progress_bar_metrics = apply_to_collection(self._progress_bar_metrics, *args)
        #self._callback_metrics = apply_to_collection(self._callback_metrics, *args)
            
        #self._signal_connector.teardown()


        # calling teardown hooks
        #call._call_teardown_hook(self)
        #trainer.lightning_module._metric_attributes = None

    #cleanup()

        

if __name__ == '__main__':
    run()
