import torch

def get_num_classes_of_yolox_checkpoint(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    for k, v in ckpt.items():
        if "cls_preds" in k and "weight" in k:
            return v.shape[0]
    assert False, f"Couldn't guess yolox num classes from checkpoint {ckpt_path}"