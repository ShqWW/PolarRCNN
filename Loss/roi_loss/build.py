def build_roi_loss(cfg):
    lossfun = None
    if cfg.roi_loss == 'tribranch_loss':
        from .tribranch_loss.lossfun import TriBranchLoss
        lossfun = TriBranchLoss(cfg)
    return lossfun