from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(cfg, use_cuda):
    return DeepSort(cfg.DEMO.DEEPSORT_REID_CKPT,
                max_dist=cfg.DEMO.DEEPSORT_MAX_DIST, min_confidence=cfg.DEMO.DEEPSORT_MIN_CONFIDENCE,
                nms_max_overlap=cfg.DEMO.DEEPSORT_NMS_MAX_OVERLAP, max_iou_distance=cfg.DEMO.DEEPSORT_MAX_IOU_DISTANCE,
                max_age=cfg.DEMO.DEEPSORT_MAX_AGE, n_init=cfg.DEMO.DEEPSORT_N_INIT, nn_budget=cfg.DEMO.DEEPSORT_NN_BUDGET, use_cuda=use_cuda)
    









