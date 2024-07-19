import torch
from tqdm import tqdm
from tools.get_config import get_cfg
from Dataset.build import build_testset
from Models.build import build_model
from Eval.build import build_evaluator
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='view-config')
    parser.add_argument('--gpu_no', default=0, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--cfg', default='./Config/config.py', type=str)
    parser.add_argument('--result_path', default='./result', type=str)
    parser.add_argument('--weight_path', default='', type=str)
    parser.add_argument('--view_path', default='./view', type=str)
    parser.add_argument('--is_view', default=0, type=int)
    parser.add_argument('--is_val', default=0, type=int) 
    args = parser.parse_args()
    return args

cfg = get_cfg(parse_args())
torch.cuda.set_device(cfg.gpu_no)

net = build_model(cfg)
net.load_state_dict(torch.load(cfg.weight_path, map_location='cpu'), strict=True)
net.cuda().eval()

tsset = build_testset(cfg)
print('testset length:', len(tsset))
evaluator = build_evaluator(cfg)

evaluator.pre_process()

tsloader = torch.utils.data.DataLoader(tsset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=16, 
                                           drop_last=False, collate_fn = tsset.collate_fn)

for i, (img, file_names, ori_imgs) in enumerate(tqdm(tsloader, desc=f'Model is running')):
    with torch.no_grad():
        img = img.cuda()
        outputs = net(img)
    if cfg.is_view:
        evaluator.view_output(outputs, file_names, ori_imgs)
    else:
        evaluator.write_output(outputs, file_names)

if cfg.is_view:
    evaluator.view_gt()
else:
    evaluator.evaluate()


