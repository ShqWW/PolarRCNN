def build_neck(cfg):
    if cfg.neck == 'fpn':
        from .fpn import FPN
        head = FPN(cfg)
    elif cfg.neck == 'fpn_hough':
        from .fpn_hough import FPN
        head = FPN(cfg)
    return head 