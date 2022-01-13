from os.path import join, dirname
#Options: det, vqa, cic, cap, cls see data/dataset
TRAIN_TASKS = ['det']
#options: image_contrast, text_contrast, mil, synonym
DET_LESSONS = ['mil','image_contrast','synonym','text_contrast']
#choose from list above
LESSONS = DET_LESSONS
#parallel 
PARALLEL = False 
#Det lesson params 
#image contrast
IMAGE_CONTRAST_CREATE_BATCH = 16
#batch size = create_batch *train_batch
IMAGE_CONTRAST_TRAIN_BATCH = 1
#text contrast
TEXT_COTRAST_CREATE_BATCH = 16
TEXT_CONTRAST_TRAIN_BATCH = 1
#mil 
MIL_CREATE_BATCH = 16
MIL_BATCH = 1
#synonym
SYNONYM_CREATE_BATCH = 2
SYNONYM_TRAIN_BATCH = 8
GLOBAL_OUTPUT_FILE = '/shared/rsaas/michal5/gpv_michal/outputs/auto_select/exp_2'