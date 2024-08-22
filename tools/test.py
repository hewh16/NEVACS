# Copyright (c) OpenBII
# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os,time
import warnings

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import sys

sys.path.append("../")
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from torch import ops
import numpy
import time
import copy
from mmcls.models.layers import lif,lifplus
from mmcls.models.backbones import base_backbone 

# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model
  
def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--use_lyngor', type=int, help='use lyngor flag: 1 or 0, means if use lyngor or not')
    parser.add_argument('--use_legacy', type=int, help='use legacy flag: 1 or 0, means if use legacy or not')
    parser.add_argument('--c', default=0, type=int, help='compile only flag: 1 or 0, means if compile only or not')
    parser.add_argument('--v', default=0, type=int, help='compile version flag: 1 or 0, means compile v1 or v0')
    parser.add_argument('--out', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
             f'result file. You can choose some of ({", ".join(out_options)}), '
             'or use "all" to include all above, or use "none" to disable all of '
             'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
             '"accuracy", "precision", "recall", "f1_score", "support" for single '
             'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
             'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be parsed as a dict metric_options for dataset.evaluate()'
             ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
             'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


from mmcls.models.layers import lif
from mmcls.models.layers import lifplus

lif.spike_func = lambda _: torch.gt(_, 0.).to(_.dtype)
lifplus.spike_func = lambda _: torch.gt(_, 0.).to(_.dtype)

CC_PATH = '/home/lynchip/ProgramFiles/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++'


def load_library():
    # load libcustom_op.so
    library_path = "./custom_op_in_pytorch/build/libcustom_ops.so"
    ops.load_library(library_path)

def load_config(args):    
    cfg = mmcv.Config.fromfile(args.config)
    args.metrics = 'accuracy'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    if args.use_lyngor == 1:
        base_backbone.ON_APU  = True
        base_backbone.FIT  = True
        if 'soma_params' in cfg.model["backbone"] and cfg.model["backbone"]['soma_params'] == 'channel_share':
            base_backbone.FIT = False 
    else:
        base_backbone.ON_APU = False


    model = build_classifier(cfg.model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None and torch.cuda.is_available():
        wrap_fp16_model(model)

    # if USE_LYNGOR:
    cfg.data.samples_per_gpu = 1  # gpu和apu均只有单batch    
    

    return model,cfg


def load_data(model_backbone_type,cfg, args,USE_LYNGOR,distributed):
    if "FastTextItout" in model_backbone_type and (not USE_LYNGOR):
        dataset = build_dataset(cfg.data.val)
        t = dataset.__getitem__(0)['img'].shape
        in_size = None
    elif "FastTextItout" in model_backbone_type:
        dataset = build_dataset(cfg.data.test)
        t, c = dataset.__getitem__(0)['img'].shape
        in_size = [((1, c),)]
        print("shape", t, c)
    elif "Jester20bn"== cfg.dataset_type and not USE_LYNGOR:
        dataset = build_dataset(cfg.data.test_gpu)
        t, c, h, w = dataset.__getitem__(0)['img'].shape
        in_size = [((1, c, h, w),)]
    else:
        dataset = build_dataset(cfg.data.test)
        t, c, h, w = dataset.__getitem__(0)['img'].shape
        in_size = [((1, c, h, w),)]
        t = 10
        print("shape", t, c, h, w)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True,
        drop_last=True)

    if hasattr(dataset, "CLASSES"):        
        CLASSES = dataset.CLASSES        
        if CLASSES is None:
            from mmcls.datasets import ImageNet
            CLASSES = ImageNet.CLASSES
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    if "FastTextItout" in model_backbone_type:  ############ imdb最后只输出一个数
        classes_num = 1
    else:
        classes_num = len(CLASSES)

    opath = '/'.join(os.path.abspath(args.config).split('/')[:-1])
    datasets_abbr = {"dvsmnist": "Dm",
                         "celldataset": "Ce",
                         "luna16cls": "Lc",
                         "dvsgesture": "Dg",
                         "cifar10dvs": "Cd",
                         "rgbgesture": "Rg",
                         "jester": "Jt",
                         "esimagenet": "Es",
                         "imdb":"Imdb"}
    if cfg.model["backbone"].type != 'ResNetLifItout' and cfg.model["backbone"].type != 'ResNetLifReluItout':
        config_name_list = cfg.filename.split('/')[-1].split('-')
        dataset_key = config_name_list[2].split('.')[0]
        network = config_name_list[0].capitalize() + datasets_abbr[dataset_key] 
    else:
        config_name_list = cfg.filename.split('/')[-1].split('-')        
        dataset_key = config_name_list[3].split('.')[0]
        network = config_name_list[0].capitalize() + datasets_abbr[dataset_key] + config_name_list[1].capitalize()
    _base_ = os.path.join(opath, network)
    return  data_loader, classes_num,dataset,CLASSES, in_size,t, _base_

def model_compile(dataset, model_backbone_type,model,args,cfg,_base_,in_size,version):
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)      

    cfg.fp16 = None    
    import model_operate 
    import custom_op_in_lyn.custom_op_my_lif
    load_library()    
    input_type = "uint8"
    model_operate.run(model.backbone.eval(), in_size, out_path=f'{_base_}',input_type=input_type,version=version)    


def apu_infer(model_backbone_type,model_path,data_loader,dataset,chip_id,t,classes_num):
    from lyn_sdk_model import ApuRun
    results = []
    model_path = os.path.join(model_path,"Net_0")
    arun = ApuRun(chip_id, model_path)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        data_img = numpy.array(data["img"]).astype(np.float32) 
        #startTime = time.time()             
        for ti in range(t): 
            apuLoadTime = time.time()         
            arun.load_data(data_img[:, ti, ...])
            apuStartTime = time.time()            
            arun.run_single(ti)
            apuEndTime = time.time()
            print(apuStartTime-apuLoadTime,apuEndTime-apuStartTime,apuEndTime-apuLoadTime)
        #endTime = time.time() 
        #print(endTime-startTime)
        output = arun.get_output()[:, 0:classes_num]       
        output1 = copy.deepcopy(output)
        results.append(output1)
        prog_bar.update()         
    if "FastTextItout" in model_backbone_type:
        eval_results = dataset.evaluate(torch.sigmoid(torch.tensor(results)))
    else:
        eval_results = dataset.evaluate(results)
    for k, v in eval_results.items():
        print(f'\n{k} : {v:.2f}')
    arun.release()

def cpu_infer(model,dataset,data_loader,distributed,args,CLASSES):
    # build the model and load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)

    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        gpuStartTime = time.time() 
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                    **show_kwargs)
        gpuEndTime = time.time() 
        print(gpuEndTime-gpuStartTime)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                    args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        if args.metrics:            
            eval_results = dataset.evaluate(outputs, args.metrics,
                                            args.metric_options)
            results.update(eval_results)
            for k, v in eval_results.items():
                print(f'\n{k} : {v:.2f}')
        if args.out:
            if 'none' not in args.out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    'class_scores': scores,
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class
                }
                if 'all' in args.out_items:
                    results.update(res_items)
                else:
                    for key in args.out_items:
                        results[key] = res_items[key]
            print(f'\ndumping results to {args.out}')
            mmcv.dump(results, args.out)



def main():
    os.chdir('../')
    args = parse_args()
    assert args.use_lyngor in (0, 1), 'use_lyngor must in (0, 1)'
    assert args.use_legacy in (0, 1), 'use_legacy must in (0, 1)'
    assert args.c in (0, 1), 'c must in (0, 1)'
    assert args.v in (0, 1), 'v must in (0, 1)'
    USE_LYNGOR = True if args.use_lyngor == 1 else False
    USE_LEGACY = True if args.use_legacy == 1 else False
    COMPILE_ONLY = True if args.c == 1 else False    

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    if args.checkpoint:
        args.checkpoint = './work_dirs/' + args.config + '/' + args.checkpoint  #只推理可以不加--checkpoint
    args.config = './configs/' + args.config + '.py'

    model,cfg= load_config(args)

    model_backbone_type = cfg.model["backbone"]["type"]

    
    data_loader, classes_num,dataset,CLASSES ,in_size,t, _base_= load_data(model_backbone_type, cfg, args,USE_LYNGOR,distributed)  
    if model_backbone_type == 'ResNetLifItout':
        model.backbone.timestep = t   

    if USE_LYNGOR is True:
        if not USE_LEGACY:
            model_compile(dataset, model_backbone_type,model,args,cfg,_base_,in_size,args.v)
        elif COMPILE_ONLY:
            print("no compile and no apu infer, so nothing to do")

        if cfg.get('lynxi_devices') is not None:

            chip_id = cfg.lynxi_devices[0][0]
        else:
            chip_id = 0

        if not COMPILE_ONLY:
            apu_infer(model_backbone_type,_base_,data_loader, dataset, chip_id,t,classes_num)

    else:
        cpu_infer(model, dataset,data_loader,distributed,args,CLASSES)


if __name__ == '__main__':
    main()
