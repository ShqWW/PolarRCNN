def build_rpn_loss(cfg):
    lossfun = None
    if cfg.rpn_loss == 'polarmap_loss':
        from .polar_map_loss import PolarMapLoss
        lossfun = PolarMapLoss(cfg)
    return lossfun