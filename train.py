# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom datasets.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# comet_ml是一个用于机器学习实验跟踪和协作的工具，可以用于记录和监视机器学习实验的参数、指标和结果
try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

# 用于获取YOLOv5根目录并将其添加到Python系统路径（sys.path）以便Python解释器能够找到YOLOv5的模块和文件。然后，通过计算相对路径，将YOLOv5的根目录保存在ROOT变量中，以供后续使用
# 创建了一个Path对象FILE，该对象表示当前文件的绝对路径。__file__是Python内置变量，表示当前模块的文件名。通过resolve()方法，可以获取FILE的绝对路径
FILE = Path(__file__).resolve()
# 使用parents属性获取FILE的父目录，即YOLOv5的根目录。parents[0]表示直接父目录，可以使用parents[1]表示祖父目录，依此类推
ROOT = FILE.parents[0]  # YOLOv5 root directory
# 这一行代码检查YOLOv5的根目录是否已经在Python系统路径sys.path中。sys.path是一个列表，其中包含Python解释器搜索模块的目录
if str(ROOT) not in sys.path:
    # 如果YOLOv5的根目录不在sys.path中，就将其添加到sys.path，从而使Python解释器能够在该目录中找到YOLOv5的模块和文件
    sys.path.append(str(ROOT))  # add ROOT to PATH
# 计算相对路径，并将其保存在ROOT中。os.path.relpath函数用于计算相对路径，它接受两个参数：第一个参数是要计算相对路径的目标路径（即YOLOv5的根目录），
# 第二个参数是相对于哪个目录计算路径（这里使用Path.cwd()表示当前工作目录）
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

# 获取环境变量的值并将其保存在相应的变量中，以便在后续代码中使用。环境变量通常用于在运行时配置程序的行为或获取运行环境的相关信息
# 通过os.getenv方法获取名为LOCAL_RANK的环境变量的值，并将其转换为整数。os.getenv用于获取指定名称的环境变量的值。如果没有找到名为LOCAL_RANK的环境变量，则将LOCAL_RANK的值设为-1
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
# 类似于上面的代码，这一行代码获取名为RANK的环境变量的值，并将其转换为整数。如果没有找到名为RANK的环境变量，则将RANK的值设为-1
RANK = int(os.getenv('RANK', -1))
# 获取名为WORLD_SIZE的环境变量的值，并将其转换为整数。如果没有找到名为WORLD_SIZE的环境变量，则将WORLD_SIZE的值设为1
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# 调用check_git_info()函数，该函数用于检查当前代码的Git信息（比如当前分支、提交哈希等）。它可能是为了在YOLOv5模型中记录模型的版本和Git信息，以便在需要时进行追踪和排查问题
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    # 这些变量是通过解构opt命名空间中的属性得到的。它们表示训练过程中的一些配置和选项，比如保存目录、训练轮数、批次大小、权重文件路径、是否进行单类别分类、是否进行模型自适应等等
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    '''
    目的是进行一些配置和参数的初始化，为后续的训练过程做准备
    '''
    # 在训练开始之前，会运行名为on_pretrain_routine_start的回调函数，以便在训练开始前执行一些初始化操作
    callbacks.run('on_pretrain_routine_start')

    # Directories 设置保存模型的路径
    w = save_dir / 'weights'  # weights dir
    # 如果evolve为真（表示进行模型自适应），则在weights目录的上级目录创建保存目录。否则，直接在weights目录下创建保存目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters 超参数的提取
    # 如果hyp是一个字符串（表示超参数配置文件的路径），则从该文件中加载超参数配置。否则，假设hyp是一个包含超参数的字典
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    # 将超参数hyp复制一份并保存在opt.hyp中，以便在后续训练过程中保存到检查点
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings 保存运行的设置参数
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    # 在这里先将data_dict设置为None，后续会根据一些条件给data_dict赋值
    data_dict = None
    # 这是一个条件语句，只有当RANK的值为-1或0时才会执行下面的代码块。RANK是一个环境变量，用于标识当前进程的排名，用于分布式训练时的多进程通信
    if RANK in {-1, 0}:
        # loggers类用于记录训练过程中的指标和损失，并将它们输出到控制台和保存到日志文件
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        # 用于迭代loggers对象的方法，并在callbacks中注册这些方法作为回调函数。回调函数将在训练过程中的不同阶段被调用，用于记录和处理不同的事件
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom datasets artifact link
        # 将data_dict赋值为loggers对象的remote_dataset属性。remote_dataset可能包含自定义数据集的信息，这些信息可能被用于日志记录和其他用途
        data_dict = loggers.remote_dataset
        # 如果resume为真，表示正在从之前保存的检查点文件中恢复训练，那么将使用opt中的权重、轮次、超参数和批量大小（batch size）来替换原有的权重、轮次、超参数和批量大小
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    # 这里根据evolve和opt.noplots的值来决定是否创建绘图
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    # 这是一个初始化随机种子的操作，用于保证实验的可复现性。使用opt.seed和RANK来设置随机种子
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 这是一个上下文管理器，用于处理分布式训练时的数据加载和划分。torch_distributed_zero_first的作用是将数据划分到多个进程中，并确保每个进程都获得一部分数据
    with torch_distributed_zero_first(LOCAL_RANK):
        # 检查data_dict是否为空，如果为空，则调用check_dataset(data)函数来获取数据集的信息
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    # 根据single_cls参数来确定类别数nc。如果single_cls为真，表示是单类别分类任务，将nc设置为1；否则，将nc设置为数据集中的类别数
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO datasets

    '''
    目的是根据参数配置创建模型，加载预训练模型的权重（如果有的话），并根据需要冻结模型的部分层
    '''

    # Model
    # 检查weights文件是否以.pt为后缀。check_suffix函数用于检查文件后缀是否符合预期
    check_suffix(weights, '.pt')  # check weights
    # 根据weights文件的后缀是否为.pt，判断模型是否是预训练模型
    pretrained = weights.endswith('.pt')
    if pretrained:
        # 这是一个上下文管理器，用于处理分布式训练时的数据加载和划分。torch_distributed_zero_first的作用是将数据划分到多个进程中，并确保每个进程都获得一部分数据
        with torch_distributed_zero_first(LOCAL_RANK):
            # 如果本地没有找到weights指定的预训练模型文件，则会尝试从网络上下载模型权重文件
            weights = attempt_download(weights)  # download if not found locally
        # 加载预训练模型的权重文件。map_location='cpu'表示将权重文件加载到CPU上，以避免CUDA内存泄漏
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        # 这一行代码创建了YOLOv5的模型。如果cfg参数为空，则会从预训练模型的检查点中获取模型配置；否则，将使用cfg指定的模型配置。ch=3表示输入图像通道数为3，nc=nc表示输出的类别数
        # .... anchors=hyp.get('anchors')表示锚框的参数，如果hyp中有'anchors'参数，则使用它，否则使用默认值
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # 根据条件决定要排除的键（即不加载的参数）。如果cfg或hyp中有'anchors'参数，并且不是从检查点中恢复训练，则排除'anchor'键
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        # 将预训练模型的权重转换为FP32（单精度浮点数）格式，并获取模型的状态字典csd
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # 用于对比模型的状态字典和预训练模型的状态字典，得到两者交集，并保存在csd中
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 将交集后的权重加载到模型中，使用strict=False表示允许部分权重不匹配的情况
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    # 用于检查模型是否支持混合精度训练（Automatic Mixed Precision，AMP）
    amp = check_amp(model)  # check AMP

    # Freeze
    # 根据freeze参数确定要冻结的层。freeze参数可以是一个列表或一个整数，根据不同的情况，生成要冻结的层的名称列表
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # 用于遍历模型的所有参数，并根据freeze参数决定是否冻结某些层的参数。如果某个层的参数在freeze列表中，则将其requires_grad属性设为False，从而冻结该层的参数
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    '''
    数据加载、超参数设置、模型的初始化、优化器设置等重要步骤
    '''

    # Image size
    # 计算模型的最大步长gs，作为网格大小（grid size）
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    # 检查RANK和batch_size的值。如果RANK为-1（单GPU训练）且batch_size为-1（未指定批次大小），则需要估算最佳批次大小
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        # 用于估算最佳的训练批次大小。check_train_batch_size函数将根据模型、输入图像大小和混合精度训练等因素估算合适的批次大小，并将其赋值给batch_size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({'batch_size': batch_size})

    # Optimizer
    # nbs为指定的标称批次大小（nominal batch size），用于设置梯度累积等参数
    nbs = 64  # nominal batch size
    # 计算梯度累积步数。梯度累积是指将几个小批次的梯度累积起来，等价于增大了批次大小。这可以在内存有限或批次大小受限的情况下使用较大的批次大小
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 缩放权重衰减（weight decay）的值，使其适应梯度累积
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    # 创建一个优化器，采用智能方式选择优化器类型（opt.optimizer），并设置学习率（hyp['lr0']）、动量（hyp['momentum']）和权重衰减（hyp['weight_decay']）等参数
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    # 检查是否采用余弦学习率调度。如果opt.cos_lr为真，表示采用余弦学习率调度。余弦学习率调度将学习率从初始值逐渐减小到最小值
    if opt.cos_lr:
        # 将使用one_cycle函数生成学习率调度器。该函数将学习率从1降到hyp['lrf']
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        # 不采用余弦学习率调度，将使用线性学习率调度，学习率从1降到hyp['lrf']
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # 创建学习率调度器，将使用上述生成的学习率调度函数lf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    #  创建指数移动平均模型EMA（Exponential Moving Average）。EMA用于在训练过程中维护模型参数的移动平均值，以平滑模型的更新
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        # 如果继续训练模型（即从上次的检查点恢复训练），将使用smart_resume函数加载最新的检查点，并获取起始的训练轮次和总训练轮次。
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        # 删除之前加载的检查点和状态字典，释放内存，之前已经加载过并且写入到模型已经换缓存文件中
        del ckpt, csd

    # DP mode
    # 这里检查是否在单GPU训练下，使用了多个GPU。如果使用了多个GPU并且没有采用分布式训练策略，则会在模型上使用torch.nn.DataParallel来实现数据并行
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            'WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
            'See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started.'
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    # 如果开启了同步批归一化（SyncBatchNorm），则会在模型上使用torch.nn.SyncBatchNorm来实现同步批归一化
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed) # 其中包括数据增强（augment=True）、缓存选项、排除矩形标注（rect=opt.rect）、多GPU训练的排名（rank=LOCAL_RANK）、以及其他参数的设置
    # 数据集中的标签拼接在一起，用于后续操作
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class 找出数据集中最大的标签类别索引，用于后续检查
    # 断言语句，检查数据集中的最大类别索引是否小于nc（类别总数）。如果不满足条件，将引发异常
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        # 创建验证数据加载器。类似于训练数据加载器，create_dataloader函数用于生成验证数据加载器，其中包括验证数据的处理方式、缓存选项等参数的设置
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            # 如果未指定noautoanchor选项，则运行check_anchors函数进行自动锚框（anchor）的检查。check_anchors函数用于自动计算最佳的锚框尺寸
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            # 将模型参数转换为半精度浮点数（FP16）计算，并将输出转换回单精度浮点数（FP32）
            model.half().float()  # pre-reduce anchor precision

        # 在on_pretrain_routine_end阶段运行回调函数，传递数据集的标签和类别名称
        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    # 果使用了CUDA（GPU）并且RANK不是-1（即使用了分布式训练），则通过smart_DDP函数对模型进行智能的分布式数据并行（DDP）处理。smart_DDP函数用于根据分布式训练的设置自动选择合适的DDP方式
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps) 获取模型中检测层（detection layer）的数量
    # 根据检测层的数量缩放超参数，以适应不同数量的检测层
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model 将类别数量（nc）赋值给模型的属性nc，用于后续训练过程中的处理
    model.hyp = hyp  # attach hyperparameters to model 将超参数hyp赋值给模型的属性hyp，用于训练过程中的设置
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights 计算并赋值给模型的属性class_weights，用于处理样本类别不平衡问题
    model.names = names # 将类别名称赋值给模型的属性names，用于后续输出结果时的标签显示

    # Start training
    t0 = time.time() # 记录训练开始时间
    nb = len(train_loader)  # number of batches 获取训练数据加载器的批次数量
    # 计算预热迭代的数量。hyp是超参数的字典，'warmup_epochs'指定了预热的迭代次数。nb是训练数据加载器的批次数量
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1 # 记录上一次优化更新的迭代数，默认初始化为-1
    maps = np.zeros(nc)  # mAP per class 用于存储每个类别的平均精度（mAP）
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls) 用于存储训练过程中的评估结果
    scheduler.last_epoch = start_epoch - 1  # do not move 将学习率调度器的last_epoch设置为开始的epoch数减1
    scaler = torch.cuda.amp.GradScaler(enabled=amp) # 创建一个梯度缩放器（GradScaler），用于在混合精度训练中自动缩放梯度
    stopper, stop = EarlyStopping(patience=opt.patience), False # 创建一个用于早停的EarlyStopping对象
    compute_loss = ComputeLoss(model)  # init loss class 初始化一个损失函数计算类
    callbacks.run('on_train_start') # 运行on_train_start回调函数
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start') # 运行on_train_epoch_start回调函数
        model.train() # 将模型设置为训练模式

        # Update image weights (optional, single-GPU only)
        # 如果指定了image_weights，则计算并更新图像权重，用于数据集中样本类别不平衡的情况
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # datasets.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        # 只在主进程（RANK为-1或0）执行下面的代码。主进程负责输出训练进度条和日志等
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        # 将优化器的梯度清零，准备进行新的前向传播和反向传播
        optimizer.zero_grad()
        # 遍历每个批次。i表示当前批次的索引，imgs是输入图像的张量，targets是图像的目标标签，paths是图像的路径，_用于接收其他不需要的元素
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            # 计算当前批次在整个训练过程中的综合批次数。用于计算预热的迭代数和学习率等
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 将输入图像转换为设备（GPU或CPU）上的浮点张量，并将像素值从0-255缩放到0.0-1.0
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            # 将输入图像转换为设备（GPU或CPU）上的浮点张量，并将像素值从0-255缩放到0.0-1.0
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                # 计算当前迭代的累积数量，用于实现梯度累积。在预热阶段，accumulate为1；在预热后，根据当前迭代数在总迭代数范围内进行插值计算
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                # 对优化器的不同参数进行线性插值，实现预热阶段的学习率和动量（momentum）的变化。xi定义了预热阶段的起始和结束迭代数，hyp是超参数字典，lf是学习率调度器的衰减函数
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            # 如果指定了multi_scale，则在每个批次中使用不同的图像尺寸（multi-scale训练）
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            '''
            前向传播、反向传播、优化器更新、学习率调整、验证集评估和模型保存等步骤。整个训练过程会多次执行这个 epoch 的循环，直到达到预设的训练轮数（epochs）或满足早停条件
            '''

            # Forward
            # 使用混合精度训练，自动选择最合适的数据类型
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward 模型的前向传播，得到预测结果
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size 计算损失函数
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            torch.use_deterministic_algorithms(False) # CBAM
            # 梯度缩放后进行反向传播
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # 当累积的迭代次数达到指定值时，进行优化器的更新
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients 当累积的迭代次数达到指定值时，进行优化器的更新
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients 裁剪梯度，以防止梯度爆炸
                scaler.step(optimizer)  # optimizer.step 裁剪梯度，以防止梯度爆炸
                scaler.update() # 更新缩放器状态
                optimizer.zero_grad()
                # 如果使用EMA（Exponential Moving Average）平均模型，则进行模型参数的EMA更新
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                # 如果回调函数标记需要停止训练，则返回
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers 获取当前优化器的学习率，用于日志记录
        scheduler.step() # 调度器进行一步更新，更新当前 epoch 的学习率

        # 只在主进程（RANK为-1或0）执行下面的代码。主进程负责输出验证结果、保存模型和进行早停检查
        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch) # 运行on_train_epoch_end回调函数。此函数用于在每个 epoch 结束时执行一些自定义操作
            # 更新EMA模型（Exponential Moving Average，指数移动平均）的属性，包括模型配置、类别数量、超参数、类别名称、步长和类别权重等
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                # 行验证集的评估。此处调用了validate.run函数，用于计算验证集的mAP和其他指标。得到验证结果results、每个类别的mAP值maps和一些其他指标
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # Update best mAP 更新最佳 mAP：将当前的验证结果与历史最佳 mAP 进行比较，更新最佳 mAP 和相应的模型权重
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model 保存模型：根据设定的条件，保存当前 epoch 的模型权重。同时保存最佳模型和最后一个 epoch 的模型
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping 如果启用了早停检查（EarlyStopping），会根据当前的 mAP 和设定的早停条件决定是否停止训练
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks 如果触发了早停，跳出整个训练过程

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        # 计算整个训练过程所用的时间，并将其打印出来
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        # 针对最后一个 epoch 和最佳模型（best），进行以下操作
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers 去除优化器的信息，只保留模型的权重信息
                # 如果是最佳模型，则进行验证集评估，计算验证结果并保存结果。如果验证集是 COCO 数据集，则同时生成 COCO 格式的结果文件（json 文件）
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        # 运行 on_fit_epoch_end 回调函数，传递模型训练的损失和验证集结果等参数
        callbacks.run('on_train_end', last, best, epoch, results) # 运行 on_train_end 回调函数，表示整个训练过程已经结束

    # 释放 GPU 缓存空间
    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / r'weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='datasets.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of datasets artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks 首先进行一些前期的检查。如果是单 GPU 训练或者是主进程（RANK为0）
    if RANK in {-1, 0}:
        print_args(vars(opt)) # 首先进行一些前期的检查。如果是单 GPU 训练或者是主进程（RANK为0）
        check_git_status() # 首先进行一些前期的检查。如果是单 GPU 训练或者是主进程（RANK为0）
        check_requirements(ROOT / 'requirements.txt') # 检查是否安装了训练所需的 Python 库，要求这些库在文件 requirements.txt 中列出

    # Resume (from specified or most recent last.pt) 这部分代码处理模型的恢复训练（Resume）或者正常训练的情况
    # 如果 opt.resume 为 True，并且不是进化训练，则尝试从指定的 opt.resume 文件中恢复训练。如果 opt.resume 是字符串，表示指定了一个 .pt 文件路径；
    # 否则，获取最近的一个 .pt 文件作为恢复的训练文件。然后，将该训练文件的配置加载到 opt 中
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original datasets
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    # 如果不是恢复训练，则对指定的 opt.data、opt.cfg、opt.hyp、opt.weights 和 opt.project 进行检查，确保这些文件或路径的有效性
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # 如果是进化训练（opt.evolve 为 True），则检查是否指定了进化的项目目录（opt.project），如果没有则默认设置为 'runs/evolve' 目录。
        # 然后，将 opt.resume 设置为 False，表示进化训练不会恢复之前的训练。
        # 同时，将模型的保存目录设置为 opt.project/opt.name，其中 opt.name 会根据 opt.cfg 文件的名称来命名
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 如果是多 GPU（DDP）训练，则执行以下操作
    if LOCAL_RANK != -1:
        # 对一些 DDP 训练的相关参数进行检查，确保合理性，比如不能同时使用 --image-weights 和 --evolve，如果使用了 -1 的 --batch-size 参数，则需要提供一个有效的 --batch-size
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    # Train
    # 如果不是进化训练，则进行正常的模型训练。调用 train 函数进行训练，传入 opt.hyp、opt、device 和 callbacks 参数
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit) 定义了一个名为 meta 的字典，用于指定超参数的进化范围
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        # 从 opt.hyp 文件中加载超参数配置到 hyp 字典中，如果在 hyp 字典中没有找到 anchors 参数，则设置默认值为 3
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        # 如果设置了 --noautoanchor 参数，则删除 hyp 字典中的 anchors 参数
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        # 设置一些训练相关的参数，如 opt.noval 和 opt.nosave 都设为 True，表示只进行最终一个 epoch 的验证和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices 定义了两个保存进化相关信息的文件：evolve_yaml 和 evolve_csv
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        # 如果指定了 --bucket 参数，则从指定的 Google Cloud Storage 存储桶中下载 evolve.csv 文件
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run([
                'gsutil',
                'cp',
                f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv), ])

        # 循环进行进化训练
        for _ in range(opt.evolve):  # generations to evolve
            # 如果 evolve.csv 文件存在，则从中选择最优的超参数并进行变异（选择或组合），生成新的一组超参数。选
            # 择父代超参数的方法有两种：一种是 single，随机选择一个父代；另一种是 weighted，按照适应度对父代进行加权选择
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                # 对新生成的一组超参数进行变异（Mutate），变异率 mp 设置为 0.8，标准差 s 设置为 0.2。
                # 变异的方式是将每个超参数的值乘以一个随机数，然后再乘以一个变异率 s，并且保持在一定的范围内。
                # 每次变异后，都检查是否有重复的超参数，如果有，则重新变异直至没有重复
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            # 将新生成的超参数限制在预定义的范围内，保留五位小数
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            # 使用新生成的超参数进行训练，并保存相关信息
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results 进化训练结束后，绘制结果并输出相应信息
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


# run 函数是一个便捷的包装器，用于使用特定的配置训练 YOLOv5 模型。它接受关键字参数（**kwargs），用于覆盖或设置特定的训练选项
def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    # 使用 parse_opt 函数解析默认的训练选项，参数为 True，表示仅解析默认参数。该函数将默认的训练选项返回为一个 opt 对象
    opt = parse_opt(True)
    # 遍历提供的关键字参数（kwargs），并将每个键值对设置为 opt 对象中的属性。这一步允许自定义选项覆盖默认的训练选项
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
