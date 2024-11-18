import torch
from .MSFN import MSFN

def model_generator(method, pretrained_model_path=None):
    if method == 'MSFN':
        model = MSFN(stage=2).cuda()


    else:
        print(f'Method {method} is not defined !!!!')
    # 读取.pth文件
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
