from .resnet import build_resnet


def create_backbone(cfg):
    if 'resnet' in cfg.TYPE:
        return build_resnet(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')