# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from diffae.experiment import LitModel
from diffae.templates import ffhq128_autoenc_w_classifier
import tqdm
from colat.models import NonlinearConditional
from colat.loss import ContrastiveLoss
from colat.projectors import IdentityProjector







## Instantiate Generator (in our case the Encoder)
conf = ffhq128_autoenc_w_classifier()
conf.include_classifier = False
conf.name = 'ffhq128_autoenc_w_classifier2'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
generator = LitModel(conf)
state = torch.load('/projects/deepdevpath2/Saranga/diffae/checkpoints/ffhq128_w_newclassifier2/last.ckpt', map_location='cpu')
generator.load_state_dict(state['state_dict'])
generator.model.eval()
generator.model.to(device)
generator.model.eval()
generator.ema_model.to(device)
# Choose mode
use_ema = False
generator = generator.ema_model if use_ema else generator.model



## Instantiate the direction model
model = NonlinearConditional(k = 100, size = 512, depth = 3)


## Instantiate the projector
projector = IdentityProjector()


## Loss function
loss_fn = ContrastiveLoss(k = 100)








# Dataset and DataLoader
train_data = conf.make_dataset(split = 'train')
test_data = conf.make_dataset(split = 'test')
print(len(train_data))
print(len(test_data))

print(train_data)
# print(data[0])
train_dataloader = conf.make_loader(train_data, shuffle = True, drop_last = True)
test_dataloader = conf.make_loader(test_data, shuffle = True, drop_last = True)


# Just to check the shapes
for batch in train_dataloader:
    imgs, idxs = batch['img'].to(device), batch['index'].to(device)
    print(imgs.shape)
    print(idxs.shape)

    cond = generator.encode(imgs)
    cond = generator.classifier_component(imgs, cond)
    print(cond.shape)
    break


# Train loop

# def train_loop(epoch, iterations):
#     pbar = tqdm.tqdm(total=iterations, leave=False)
#     pbar.set_description(f"Epoch {epoch} | Train")

 