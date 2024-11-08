def build_evaluator(cfg):
    if cfg.dataset == 'culane':
        from .culane_evaluator import CULaneEvaluator
        evaluator = CULaneEvaluator(cfg=cfg)

    elif cfg.dataset == 'llamas':
        from .llamas_evaluator import LLAMASEvaluator
        evaluator = LLAMASEvaluator(cfg=cfg)

    elif cfg.dataset == 'tusimple':
        from .tusimple_evaluator import TuSimpleEvaluator
        evaluator = TuSimpleEvaluator(cfg=cfg)

    elif cfg.dataset == 'dlrail':
        from .dlrail_evaluator import DLRailEvaluator
        evaluator = DLRailEvaluator(cfg=cfg)

    elif cfg.dataset == 'curvelanes':
        from .curvelanes_evaluator import CurveLanesEvaluator
        evaluator = CurveLanesEvaluator(cfg=cfg)
        
    else:
        evaluator = None
    return evaluator