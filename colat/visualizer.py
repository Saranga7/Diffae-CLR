import logging
import math
import random
from typing import List, Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as T
import tqdm
from PIL import Image, ImageDraw, ImageFont
from omegaconf.listconfig import ListConfig

# from colat.generators import Generator
from diffae.experiment import LitModel

sign = lambda x: math.copysign(1, x)


class Visualizer:
    """Model evaluator

    Args:
        model: model to be evaluated
        generator: pretrained generator
        projector: pretrained projector
        device: device on which to evaluate model
        n_samples: number of samples
    """

    def __init__(
        self,
        model: torch.nn.Module,
        generator: LitModel,
        projector: torch.nn.Module,
        device: torch.device,
        n_samples: Union[int, str],
        n_dirs: Union[int, List[int]],
        alphas: List[int],
        iterative: bool = True,
        feed_layers: Optional[List[int]] = None,
        image_size: Optional[Union[int, List[int]]] = None,
        conf = None,
        mode: str = 'non_ema'
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Diffae conf
        self.conf = conf
        self.mode = mode

        # Device
        self.device = device

        # Model
        self.model = model
        self.generator = generator
        self.projector = projector

        # Set to eval
        self.generator.eval()
        self.projector.eval()
        self.model.eval()

        assert type(n_samples) is int

        # N Samples
        # self.samples = self.generator.sample_latent(n_samples)
        # self.samples = self.samples.to(self.device)

        test_data = self.conf.make_dataset(split = 'test')
        # Get the total number of samples in the dataset
        dataset_length = len(test_data)
        # Ensure n_samples does not exceed the dataset size
        n_samples = min(n_samples, dataset_length)
        # Generate n_samples random indices
        random_indices = random.sample(range(dataset_length), n_samples)
        # Sample the images using the generated indices
        sampled_images = [test_data[idx]['img'] for idx in random_indices]
        # Convert the list of sampled images to a tensor (if necessary)
        self.sampled_images = torch.stack(sampled_images).to(self.device)
        # print(self.sampled_images.shape)

        # exit()


        #  Sub-sample Dirs
        if n_dirs == -1:
            self.dirs = list(range(self.model.k))
        elif isinstance(n_dirs, int):
            self.dirs = np.random.choice(self.model.k, n_dirs, replace=False)
        else:
            assert isinstance(n_dirs, ListConfig)
            self.dirs = n_dirs

        # Alpha
        alphas = sorted(alphas)
        i = 0
        while alphas[i] < 0:
            i += 1
        self.neg_alphas = alphas[:i]

        if alphas[i] == 0:
            i += 1
        self.pos_alphas = alphas[i:]

        # Iterative
        self.iterative = iterative

        # Image Size
        if image_size:
            self.image_transform = T.Resize(image_size)
        else:
            self.image_transform = torch.nn.Identity()

        # Feed Layers
        self.feed_layers = feed_layers


    def visualize(self) -> float:
        """Generates images from the trained model with classifier outputs displayed.

        Returns:
            (float) accuracy (on a 0 to 1 scale)
        """
        # Progress bar
        pbar = tqdm.tqdm(total=self.sampled_images.shape[0], leave=False)
        pbar.set_description("Generating... ")

        # Set to eval
        self.generator.eval()
        self.generator.model.eval()
        self.generator.ema_model.eval()
        self.projector.eval()
        self.model.eval()

        def _edit(z, alpha, ks):
            assert z.shape[0] == 1 or z.shape[0] == len(
                ks
            ), "Only able to apply all directions to single latent code or apply each direction to single code"
            self.model.alpha = alpha

            # Apply Directions
            zs = []
            for i, k in enumerate(ks):
                _i = i if z.shape[0] > 1 else 0
                zs.append(self.model.forward_single(z[_i : _i + 1, ...], k=k))
            zs = torch.cat(zs, dim=0)
            return zs

        def _encode_and_concatenate_classifier(imgs):
            w = self.generator.encode(imgs, mode=self.mode)
            if self.mode == 'ema':
                concat_classifier = self.generator.ema_model.classifier_component
            else:
                concat_classifier = self.generator.model.classifier_component
            z_sem = concat_classifier(imgs, w)
            return z_sem

        def _classify(imgs, mode=self.mode):
            if mode == 'ema':
                model = self.generator.ema_model
            else:
                model = self.generator.model

            return torch.softmax(model.classifier_component.mobile_net(imgs), dim=1)

        def _generate(z_sem, xT, T=20):
            images = self.generator.render(xT, z_sem, T=T, mode=self.mode)
            return images

        # Loop
        with torch.no_grad():
            for i in range(self.sampled_images.shape[0]):
                # Take a single sample
                orj_img = self.sampled_images[i : i + 1, ...]
                z_sem = _encode_and_concatenate_classifier(orj_img)

                xT = self.generator.encode_stochastic(orj_img, z_sem, T=50, mode=self.mode)

                images = []
                classifier_outputs = []  # To store classifier outputs

                z_sem_orig = z_sem
                prev_alpha = 0
                for alpha in reversed(self.neg_alphas):
                    _z = z_sem if self.iterative else z_sem_orig
                    _alpha = alpha - prev_alpha if self.iterative else alpha

                    z_sem = _edit(_z, _alpha, ks=self.dirs)
                    xT_repeated = xT.repeat(z_sem.shape[0], 1, 1, 1)

                    generated_images = _generate(z_sem, xT_repeated)
                    c_output = _classify(generated_images)  # Get classifier output
                    classifier_outputs.append(c_output)  # Store the classifier output
                    images.append(generated_images.detach().cpu())

                    prev_alpha = alpha

                images = list(reversed(images))
                classifier_outputs = list(reversed(classifier_outputs))

                z_sem = z_sem_orig
                prev_alpha = 0
                for alpha in self.pos_alphas:
                    _z = z_sem if self.iterative else z_sem_orig
                    _alpha = alpha - prev_alpha if self.iterative else alpha

                    z_sem = _edit(_z, _alpha, ks=self.dirs)
                    generated_images = _generate(z_sem, xT_repeated)
                    c_output = _classify(generated_images)
                    classifier_outputs.append(c_output)  # Store the classifier output
                    images.append(generated_images.detach().cpu())

                    prev_alpha = alpha

                images = torch.stack(images, dim=0).transpose(1, 0)
                orj_img = (orj_img + 1)/2
                orj_img = orj_img.detach().cpu()
                col_orj_img = orj_img.repeat((images.size(0), 1, 1, 1))

                titles = []
                classifier_texts = []  # To store classifier text strings

                before_sign = -1
                imgs = []
                for ind, alpha in enumerate(self.neg_alphas + self.pos_alphas):
                    if sign(alpha) != before_sign:
                        imgs.append(col_orj_img)
                        titles.append("Original")
                        classifier_texts.append(f"{classifier_outputs[ind][0].cpu().numpy().round(2)}")  # Store classifier output for the original image
                        before_sign = sign(alpha)

                    titles.append(f"α={alpha:.1f}")
                    c_output_text = f"{classifier_outputs[ind][0].cpu().numpy().round(2)}"  # Get the specific classifier output for this image
                    classifier_texts.append(c_output_text)  # Store classifier output text for each subplot
                    imgs.append(images[:, ind, ...])
                
                images = torch.stack(imgs).transpose(1, 0)
                images = images.reshape(-1, images.size(-3), images.size(-2), images.size(-1))

                # Matplotlib figure
                fig, axes = plt.subplots(images.size(0) // len(titles), len(titles), figsize=(20, 40))  # Increased height
                if images.size(0) // len(titles) == 1:
                    axes = np.expand_dims(axes, 0)  # Ensure axes is always 2D

                for row, img_row in enumerate(images.chunk(images.size(0) // len(titles), dim=0)):
                    for col, (ax, img, title, classifier_text) in enumerate(zip(axes[row], img_row, titles, classifier_texts)):
                        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
                        ax.set_title(f"{title}", fontsize=12, pad=15)  # Increased font size and padding
                        ax.text(0.5, -0.1, f"{classifier_text}", fontsize=10, ha='center', va='top', transform=ax.transAxes)
                        ax.axis('off')

                # Adjust layout and spacing
                plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjusted for more space between images
                plt.savefig(f"sample_{i}.png", bbox_inches='tight')
                plt.close(fig)

                pbar.update()

        pbar.close()



    # def visualize(self) -> float:
    #     """Generates images from the trained model

    #     Returns:
    #         (float) accuracy (on a 0 to 1 scale)

    #     """
    #     # Progress bar
    #     pbar = tqdm.tqdm(total=self.sampled_images.shape[0], leave=False)
    #     pbar.set_description("Generating... ")

    #     # Set to eval
    #     self.generator.eval()
    #     self.generator.model.eval()
    #     self.generator.ema_model.eval()
    #     self.projector.eval()
    #     self.model.eval()

    #     #  Helper function to edit latent codes
    #     def _edit(z, alpha, ks):
    #         #  check if only one latent code is given
    #         assert z.shape[0] == 1 or z.shape[0] == len(
    #             ks
    #         ), """Only able to apply all directions to single latent code or
    #             apply each direction to single code"""
    #         self.model.alpha = alpha

    #         # Apply Directions
    #         zs = []
    #         for i, k in enumerate(ks):
    #             _i = i if z.shape[0] > 1 else 0
    #             zs.append(self.model.forward_single(z[_i : _i + 1, ...], k=k))
    #         zs = torch.cat(zs, dim=0)
    #         return zs
        
    #     def _encode_and_concatenate_classifier(imgs):
    #         w = self.generator.encode(imgs, mode = self.mode)
    #         if self.mode == 'ema':
    #             concat_classifier = self.generator.ema_model.classifier_component
    #         else:
    #             concat_classifier = self.generator.model.classifier_component
    #         z_sem = concat_classifier(imgs, w)
    #         return z_sem
        
    #     def _classify(imgs, mode = self.mode):
    #         if mode == 'ema':
    #             model = self.generator.ema_model
    #         else:
    #             model = self.generator.model

    #         return torch.softmax(model.classifier_component.mobile_net(imgs), dim=1)



    #     def _generate(z_sem, xT, T = 20):
    #         images = self.generator.render(xT, z_sem, T = T, mode = self.mode)
    #         return images
        



    #     #Loop
    #     with torch.no_grad():
    #         for i in range(self.sampled_images.shape[0]):
    #             # Take a single sample
    #             orj_img = self.sampled_images[i : i + 1, ...]
    #             z_sem = _encode_and_concatenate_classifier(orj_img)
 
    #             print("Encoding noise map of the original image")
    #             xT = self.generator.encode_stochastic(orj_img, z_sem, T = 50, mode = self.mode)


    #             images = []
    #             # classifier_outputs = []

    #             #  Start with z and alpha = 0
    #             z_sem_orig = z_sem
    #             prev_alpha = 0
    #             for alpha in reversed(self.neg_alphas):
    #                 #  if iterative use last z and d(alpha)
    #                 _z = z_sem if self.iterative else z_sem_orig
    #                 _alpha = alpha - prev_alpha if self.iterative else alpha

    #                 z_sem = _edit(_z, _alpha, ks=self.dirs)

    #                 xT_repeated = xT.repeat(z_sem.shape[0], 1, 1, 1)
    #                 # print("Generating images...")
    #                 generated_images = _generate(z_sem, xT_repeated)
    #                 # c_output = _classify(generated_images)
    #                 # classifier_outputs.append(c_output)
    #                 images.append(generated_images.detach().cpu())
                    

    #                 prev_alpha = alpha

    #             # Reverse images
    #             images = list(reversed(images))
    #             # classifier_outputs = list(reversed(classifier_outputs))

    #             # Reset z and alpha
    #             z_sem = z_sem_orig
    #             prev_alpha = 0
    #             for alpha in self.pos_alphas:
    #                 #  if iterative use last z and d(alpha)
    #                 _z = z_sem if self.iterative else z_sem_orig
    #                 _alpha = alpha - prev_alpha if self.iterative else alpha

    #                 z_sem = _edit(_z, _alpha, ks=self.dirs)
    #                 generated_images = _generate(z_sem, xT_repeated)
    #                 # c_output = _classify(generated_images)
    #                 # classifier_outputs.append(c_output)
    #                 images.append(generated_images.detach().cpu())

                    
    #                 prev_alpha = alpha

    #             #  Prepare final image
    #             images = torch.stack(images, dim=0).transpose(1, 0)

    #             orj_img = (orj_img + 1)/2
    #             orj_img = orj_img.detach().cpu()

    #             col_orj_img = orj_img.repeat((images.size(0), 1, 1, 1))  # .unsqueeze(1)

    #             titles = []
            
    #             before_sign = -1
    #             imgs = []
    #             for ind, alpha in enumerate(self.neg_alphas + self.pos_alphas):
    #                 # append original image
    #                 if sign(alpha) != before_sign:
    #                     imgs.append(col_orj_img)
    #                     titles.append("α=0")
    #                     before_sign = sign(alpha)

    #                 titles.append(f"α= {alpha:.3f}")

    #                 imgs.append(images[:, ind, ...])
    #             images = torch.stack(imgs).transpose(1, 0)

    #             images = images.reshape(
    #                 -1, images.size(-3), images.size(-2), images.size(-1)
    #             )

    

    #             imgs_grid = torchvision.utils.make_grid(
    #                 images,
    #                 nrow=len(self.neg_alphas) + len(self.pos_alphas) + 1,
    #                 padding=2,
    #                 pad_value=255,
    #             )

    #             fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

    #             # get a drawing context
    #             img_alpha = Image.new("RGB", (imgs_grid.size(-1), 50), (255, 255, 255))
    #             d = ImageDraw.Draw(img_alpha)

    #             # draw alpha text
    #             for ind, text in enumerate(titles):
    #                 # print(f"Drawing text: {text}, classifier: {classifier_text}")

    #                 d.multiline_text(
    #                     (40 + ind * (images.size(-2) + 2), 10),
    #                     text,
    #                     font=fnt,
    #                     fill=(0, 0, 0),
    #                 )
            
    #             # get a drawing context
    #             img_k = Image.new(
    #                 "RGB", (100, imgs_grid.size(-2) + 50), (255, 255, 255)
    #             )
    #             d = ImageDraw.Draw(img_k)

    #             #  draw direction text
    #             for ind in range(len(self.dirs)):
    #                 d.multiline_text(
    #                     (10, 100 + ind * (images.size(-1) + 2)),
    #                     f"k={self.dirs[ind]}",
    #                     font=fnt,
    #                     fill=(0, 0, 0),
    #                 )

    #             img_alpha = T.ToTensor()(img_alpha)
    #             img_k = T.ToTensor()(img_k)

    #             imgs_grid = torch.cat([img_alpha, imgs_grid], dim=-2)
    #             imgs_grid = torch.cat([img_k, imgs_grid], dim=-1)

    #             torchvision.utils.save_image(imgs_grid, f"sample_{i}.png")

    #             # Update progress bar
    #             pbar.update()

    #     pbar.close()
