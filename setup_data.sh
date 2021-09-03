BUCKET="https://ai2-prior-gpv.s3-us-west-2.amazonaws.com/public"

# set to the location where you want to store downloaded data (data_dir)
GPV_DATA=~/data/gpv

# download and extract coco and coco-sce splits
mkdir -p $GPV_DATA
mkdir $GPV_DATA/learning_phase_data
wget $BUCKET/coco_original_and_sce_splits.zip -P $GPV_DATA/learning_phase_data/
unzip $GPV_DATA/learning_phase_data/coco_original_and_sce_splits.zip -d $GPV_DATA/learning_phase_data/
wget $BUCKET/refcocop.zip -P $GPV_DATA/learning_phase_data/
unzip $GPV_DATA/learning_phase_data/refcocop.zip -d $GPV_DATA/learning_phase_data/

# download detr pretrained on coco and coco-sce
mkdir $GPV_DATA/detr
wget $BUCKET/detr/detr_coco.pth -P $GPV_DATA/detr/
wget $BUCKET/detr/detr_coco_sce.pth -P $GPV_DATA/detr/

# Download coco images
python -m data.coco.download download_coco_images_only=True download_coco_test_images=True output_dir=$GPV_DATA

# Download vinvl, file_paths.PRECOMPUTED_FEATURES_DIR points to "data-cache/precomputed-features"
mkdir -p data-cache
mkdir -p data-cache/precomputed-features
mkdir -p data-cache/precomputed-features/coco
wget https://ai2-prior-gpv.s3.us-west-2.amazonaws.com/precomputed-image-features/coco/vinvl.hdf5 -O data-cache/precomputed-features/coco/vinvl.hdf5