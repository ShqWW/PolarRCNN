def build_roi_head(cfg):
    if cfg.roi_head == 'global_polar_head':
        from .global_polar_head import GlobalPolarHead
        head = GlobalPolarHead(cfg)
    return head 
