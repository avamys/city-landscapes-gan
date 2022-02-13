# DCGAN implementation

Simple DCGAN basic implementation project.

## Train

Customize options in config file in and run:
```
python train_gan.py path/to/config/file path/to/image/folder
```

If your dataset is not an image folder, either use another torch Dataset type or create your own in get_loader.py

## Generate

With trained model generate images by:
```
python generate.py path/to/config/file path/to/image/folder number-to-generate
```

## Example

Below is an example of generated images from the model trained on noisy, 6x augmented city landscapes dataset

![generated-image-grid](https://i.imgur.com/NQAnsNq.png)