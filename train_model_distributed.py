import pickle
from torchvision import datasets, models, transforms
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning.metrics.functional.classification import f1_score
import itertools
from pytorch_lightning.callbacks import ModelCheckpoint
import imgaug as ia
import imgaug.augmenters as iaa
import os
from pandas.core.common import flatten

img_paths_train = []
mask_paths_train = []
img_labels_train = []

all_paths_dir = 'blend_dataset/'
all_paths_filenames = []
for root, dirs, files in os.walk(all_paths_dir):
    for file in files:
        if file.endswith(".pkl"):
            all_paths_filenames.append(os.path.join(root, file))

img_paths_train = []
mask_paths_train = []
img_labels_train = []
for train_path in all_paths_filenames[:-1]:
    with open(train_path, 'rb') as alp:
        all_paths_splits = pickle.load(alp)
        img_paths_train.append(all_paths_splits['image paths'])
        mask_paths_train.append(all_paths_splits['mask paths'])
        img_labels_train.append(all_paths_splits['image labels'])

img_paths_train = list(flatten(img_paths_train))
mask_paths_train = list(flatten(mask_paths_train))
img_labels_train = list(flatten(img_labels_train))

with open(all_paths_filenames[-1], 'rb') as alp:
    all_paths_splits = pickle.load(alp)
    img_paths_valid = all_paths_splits['image paths']
    mask_paths_valid = all_paths_splits['mask paths']
    img_labels_valid = all_paths_splits['image labels']

print("Total training samples : ", len(img_paths_train))
print("Total validation samples : ", len(img_paths_valid))


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, img_paths, mask_paths, img_labels):
        'Initialization'
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_labels = img_labels
        self.transform_img = transforms.Compose([
            transforms.Resize((528,528)),
            #transforms.CenterCrop(528),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((528,528)),
            #transforms.CenterCrop(528),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        image = self.transform_img(Image.open(self.img_paths[index]))
        mask = self.transform_mask(Image.open(self.mask_paths[index]).convert('L'))

        if self.img_labels[index] == 'fake':
            value = 1
        else:
            value = 0

        return image, mask, value

train_dataset = Dataset(img_paths_train, mask_paths_train, img_labels_train)
valid_dataset = Dataset(img_paths_valid, mask_paths_valid, img_labels_valid)
train = DataLoader(train_dataset, batch_size=6, num_workers=os.cpu_count(),
        drop_last=True)
valid = DataLoader(valid_dataset, batch_size=6, num_workers=os.cpu_count(),
        drop_last=True)

class Classifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # Resnet config
        aux_params=dict(
            pooling='max',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        self.model = smp.DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights="imagenet", in_channels=3, classes=1, aux_params=aux_params)

    def forward(self, image):
        output_mask, output_target = self.model(image)
        return output_mask, output_target

    def training_step(self, batch, batch_idx):
        image, mask, target = batch
        output_mask, output_target = self.model(image) #F.interpolate(self.resnet(image), size=64)
        mask_loss = F.mse_loss(output_mask, mask)
        #bce_loss = nn.BCEWithLogitsLoss()
        class_loss = F.binary_cross_entropy_with_logits(output_target.squeeze(), target.type(torch.DoubleTensor).cuda())
        loss = mask_loss + class_loss
        result_dict = {
            'class_loss': class_loss,
            'mask_loss': mask_loss,
            'predictions': output_target.squeeze(),
            'targets': target,
            'total_loss': loss
        }
        return {'loss': loss, 'result': result_dict}

    def validation_step(self, batch, batch_idx):
        image, mask, target = batch
        output_mask, output_target = self.model(image) #F.interpolate(self.resnet(image), size=64)
        mask_loss = F.mse_loss(output_mask, mask)
        #print('Mask Loss:', mask_loss)
        class_loss = F.binary_cross_entropy_with_logits(output_target.squeeze(), target.type(torch.DoubleTensor).cuda())
        loss = mask_loss + class_loss
        result_dict = {
            'class_loss': class_loss,
            'mask_loss': mask_loss,
            'predictions': output_target.squeeze(),
            'targets': target,
            'total_loss': loss
        }
        return result_dict

    def training_epoch_end(self, train_outputs):
        '''
        Log all the values after the end of the epoch.
        '''
        outputs = [x['result'] for x in train_outputs]

        avg_class_loss = torch.stack([x['class_loss'] for x in outputs]).mean()
        avg_mask_loss = torch.stack([x['mask_loss'] for x in outputs]).mean()
        avg_loss = torch.stack([x['total_loss'] for x in outputs]).mean()

        all_predictions = torch.stack(
            [x['predictions'] for x in outputs]).flatten()
        all_targets = torch.stack([x['targets'] for x in outputs]).flatten()

        class_accuracy = accuracy(all_predictions, all_targets, num_classes=2)
        class_f1 = f1_score(all_predictions, all_targets, num_classes=2)

        self.log('train_class_loss', avg_class_loss, sync_dist=True)
        self.log('train_mask_loss', avg_mask_loss, sync_dist=True)
        self.log('train_loss', avg_loss, sync_dist=True)
        self.log('train_accuracy', class_accuracy, prog_bar=True, sync_dist=True)
        self.log('train_f1', class_f1, sync_dist=True)

    def validation_epoch_end(self, outputs):
        '''
        Log all the values after the end of the epoch.
        '''

        avg_class_loss = torch.stack([x['class_loss'] for x in outputs]).mean()
        avg_mask_loss = torch.stack([x['mask_loss'] for x in outputs]).mean()
        avg_loss = torch.stack([x['total_loss'] for x in outputs]).mean()

        all_predictions = torch.stack(
            [x['predictions'] for x in outputs]).flatten()
        all_targets = torch.stack([x['targets'] for x in outputs]).flatten()

        class_accuracy = accuracy(all_predictions, all_targets, num_classes=2)
        class_f1 = f1_score(all_predictions, all_targets, num_classes=2)

        self.log('valid_class_loss', avg_class_loss, sync_dist=True)
        self.log('valid_mask_loss', avg_mask_loss, sync_dist=True)
        self.log('valid_loss', avg_loss, sync_dist=True)
        self.log('valid_accuracy', class_accuracy, prog_bar=True, sync_dist=True)
        self.log('valid_f1', class_f1, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=1e-3,
            step_size_up=3000,
            step_size_down=3000,
            mode='triangular2',
            cycle_momentum=False
        )
        return [optimizer], [scheduler]

# init model
model = Classifier()
checkpoint_callback = ModelCheckpoint(
    monitor='valid_loss', dirpath='checkpoints/')

# Initialize a trainer
trainer = pl.Trainer(gpus=8, accelerator='ddp', max_epochs=15, progress_bar_refresh_rate=20,
                     precision=16, callbacks=[checkpoint_callback])

# Train the model ⚡
trainer.fit(model, train, valid)

#Save and checkpoint
trainer.save_checkpoint("final_model.ckpt")
