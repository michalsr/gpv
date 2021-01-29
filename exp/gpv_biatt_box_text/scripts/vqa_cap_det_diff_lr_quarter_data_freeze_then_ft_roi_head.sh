EXP_NAME=quarter_data_gpv_biatt_det_vqa_cap_wt_cap_5e-2_roi_head_freeze_then_ft
python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=1 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10004' \
    learning_datasets=det_vqa_cap \
    training.ckpt=null \
    training.freeze=True \
    training.frozen_epochs=5 \
    training.frozen_batch_size=64 \
    training.batch_size=120 \
    training.run_eval_at_launch=False \
    training.run_vis_at_launch=False \
    training.num_workers=10 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1 \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304 \
    task_configs.coco_captioning.max_samples.train=73233 \
    task_configs.coco_detection.max_samples.train=43634 \
    task_configs.coco_vqa.max_samples.train=84597

ckpt="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}/ckpts/model.pth"
python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=1 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10004' \
    learning_datasets=det_vqa_cap \
    training.ckpt=$ckpt \
    training.freeze=False \
    training.batch_size=40 \
    training.num_epochs=20 \
    training.run_eval_at_launch=False \
    training.run_vis_at_launch=False \
    training.num_workers=10 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1 \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304 \
    task_configs.coco_captioning.max_samples.train=73233 \
    task_configs.coco_detection.max_samples.train=43634 \
    task_configs.coco_vqa.max_samples.train=84597
