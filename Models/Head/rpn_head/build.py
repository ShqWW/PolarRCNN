def build_rpn_head(cfg):
    if cfg.rpn_head == 'local_polar_head':
        from .local_polar_head import LocalPolarHead
        head = LocalPolarHead(cfg)
    elif cfg.rpn_head == 'static_anchor_head':
        from .static_anchor_head import FixAnchorHead
        head = FixAnchorHead(cfg)
    return head 
