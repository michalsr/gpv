EXP_NAME=gpv_biatt_det_vqa_cap_wt_cap_5e-2_bs_120_freeze_then_ft
python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=2 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10003' \
    learning_datasets=det_vqa_cap \
    training.ckpt=null \
    training.freeze=True \
    training.frozen_epochs=10 \
    training.frozen_batch_size=120 \
    training.batch_size=120 \
    training.num_workers=30 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1

ckpt="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}/ckpts/model.pth"
python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=2 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10003' \
    learning_datasets=det_vqa_cap \
    training.ckpt=$ckpt \
    training.freeze=False \
    training.batch_size=120 \
    training.num_workers=30 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1