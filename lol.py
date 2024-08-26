import matplotlib.pyplot as plt
from diffae.templates import ffhq128_autoenc_w_classifier
import random

# Your existing code
conf = ffhq128_autoenc_w_classifier()

train_data = conf.make_dataset(split='train')
test_data = conf.make_dataset(split='test')

print(f"Training set size: {len(train_data)}")
print(f"Testing set size: {len(test_data)}")

n_samples = 5

dataset_length = len(test_data)
# Ensure n_samples does not exceed the dataset size
n_samples = min(n_samples, dataset_length)
# Generate n_samples random indices
random_indices = random.sample(range(dataset_length), n_samples)
# Sample the images using the generated indices
sampled_images = [test_data[idx]['img'] for idx in random_indices]

# Display the sampled images
plt.figure(figsize=(15, 5))
for i, img in enumerate(sampled_images):
    plt.subplot(1, n_samples, i + 1)
    # Remove the batch dimension if necessary
    if img.ndimension() == 4 and img.size(0) == 1:
        img = img.squeeze(0)
        # img = (img + 1) / 2
    # Convert the image to numpy and transpose the dimensions to HWC
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis('off')
plt.savefig('sampled_images.png')
plt.show()



 