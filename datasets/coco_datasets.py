import os
import random
import hydra
import numpy as np
import skimage.io as skio
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T

import utils.io as io
from utils.detr_misc import collate_fn as detr_collate_fn
from .coco_generic_dataset import GenericCocoDataset


class CocoCaptioning(GenericCocoDataset):
    def __init__(self,cfg,subset):
        super().__init__(cfg,subset)

class CocoDetection(GenericCocoDataset):
    def __init__(self,cfg,subset):
        super().__init__(cfg,subset)
        if subset=='train':
            self.transforms = T.Compose([
                T.ToPILImage(mode='RGB'),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

class CocoVqa(GenericCocoDataset):
    def __init__(self,cfg,subset):
        super().__init__(cfg,subset)

    def __getitem__(self,i):
        sample = self.samples[i]

        if self.cfg.read_image is True:
            image_subset = sample['image']['subset']
            image_id = sample['image']['image_id']
            img, original_image_size = self.read_image(
                image_subset,image_id)
            img = (255*img).astype(np.uint8)
            img = self.transforms(img)
        
        query = sample['query']
        all_answers = []
        for answer,freq in sample['all_answers'].items():
            if freq > 2:
                all_answers.extend([answer]*freq)

        targets = {'answer': random.choice(all_answers)}

        if self.cfg.read_image is True:
            return img, query, targets
        else:
            return query,targets

class CocoClassification(GenericCocoDataset):
    def __init__(self,cfg,subset):
        super().__init__(cfg,subset)
        if subset=='train':
            self.transforms = T.Compose([
                T.ToPILImage(mode='RGB'),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def read_image(self,image_subset,image_id,x,y,w,h):
        img_dir = os.path.join(self.cfg.image_dir,image_subset)
        img_path = os.path.join(
            img_dir,
            'COCO_'+image_subset+'_'+str(image_id).zfill(12)+'.jpg')
        img = skio.imread(img_path)
        if len(img.shape)==2:
            img = np.tile(np.expand_dims(img,2),(1,1,3))
        else:
            img = img[:,:,:3]

        H,W = img.shape[:2]
        if w<5: w=5
        if h<5: h=5
        x1 = x - 0.2*w
        x2 = x + 1.2*w
        y1 = y - 0.2*h
        y2 = y + 1.2*h
        x1,x2 = [min(max(0,int(z)),W) for z in [x1,x2]]
        y1,y2 = [min(max(0,int(z)),H) for z in [y1,y2]]
        img = img[y1:y2,x1:x2]
        original_image_size = img.shape[:2] # HxW
        resized_image_size = resize(img,(self.imh,self.imw),anti_aliasing=True)
        return resized_image_size, original_image_size
    
    def __getitem__(self,i):
        sample = self.samples[i]

        if self.cfg.read_image is True:
            image_subset = sample['image']['subset']
            image_id = sample['image']['image_id']
            x,y,w,h = sample['boxes']
            img, original_image_size = self.read_image(
                image_subset,image_id,x,y,w,h)
            img = (255*img).astype(np.uint8)
            img = self.transforms(img)
        
        query = sample['query']

        targets = {'answer': sample['answer']}

        if self.cfg.read_image is True:
            return img, query, targets
        else:
            return query,targets

@hydra.main(config_path="../configs",config_name="test/coco_datasets")
def test_dataset(cfg):
    print(cfg.pretty())
    task_config_name = cfg.learning_datasets[cfg.dataset_to_test].task_config
    dataset = globals()[cfg.dataset_to_test](
        cfg.task_configs[task_config_name],'train')
    dataloader = dataset.get_dataloader(batch_size=2)
    for data in dataloader:
        import ipdb; ipdb.set_trace()


if __name__=='__main__':
    test_dataset()