import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration
import utils
from model.discriminator import Discriminator
from PIL import Image

def to_tensor(img):
    """Convert numpy array image to tensor in range [-1, 1]"""
    # Convert from HWC to CHW format
    img = img.transpose((2, 0, 1))
    # Convert to float tensor and normalize to [0, 1]
    img = torch.from_numpy(img).float() / 255.0
    # Transform from [0, 1] to [-1, 1]
    img = img * 2 - 1
    return img

def load_image(image_path, height, width):
    """Load and preprocess image"""
    image_pil = Image.open(image_path)
    # Resize or crop image to match model input size
    if image_pil.size[0] != width or image_pil.size[1] != height:
        image_pil = image_pil.resize((width, height), Image.Resampling.LANCZOS)
    image = np.array(image_pil)
    # Ensure image has 3 channels
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using device:', device)

    parser = argparse.ArgumentParser(description='Test discriminator on steganographic images')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--test-image', '-t', required=True, type=str,
                        help='The image to test (steganographic or original)')

    args = parser.parse_args()

    print('Loading options from:', args.options_file)
    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    print('Image size: {}x{}'.format(hidden_config.H, hidden_config.W))
    print('Discriminator blocks:', hidden_config.discriminator_blocks)
    print('Discriminator channels:', hidden_config.discriminator_channels)
    
    # Create discriminator
    discriminator = Discriminator(hidden_config).to(device)

    print('Loading checkpoint from:', args.checkpoint_file)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    # Load discriminator state dict
    if 'discriminator' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator'])
        print('Discriminator loaded successfully!')
    elif 'model' in checkpoint and 'discriminator' in checkpoint['model']:
        # Some checkpoints might have a 'model' key containing the state dicts
        discriminator.load_state_dict(checkpoint['model']['discriminator'])
        print('Discriminator loaded successfully from model dict!')
    else:
        print('Warning: Discriminator state not found in checkpoint. Using random weights.')
        print('Checkpoint keys:', list(checkpoint.keys()))
        if 'model' in checkpoint:
            print('Model keys:', list(checkpoint['model'].keys()))

    # Set discriminator to evaluation mode
    discriminator.eval()

    print('Loading test image from:', args.test_image)
    image = load_image(args.test_image, hidden_config.H, hidden_config.W)
    print('Image shape:', image.shape)
    
    image_tensor = to_tensor(image).to(device)
    image_tensor.unsqueeze_(0)  # Add batch dimension
    print('Image tensor shape:', image_tensor.shape)

    print('Testing discriminator...')
    with torch.no_grad():
        output = discriminator(image_tensor)
        # Apply sigmoid to get probability
        probability = torch.sigmoid(output).item()

    print('\n=== Discriminator Result ===')
    print('Raw output:', output.item())
    print('Probability of being steganographic:', probability)
    
    if probability > 0.5:
        print('Prediction: The image is steganographic (contains watermark)')
    else:
        print('Prediction: The image is original (no watermark)')


if __name__ == '__main__':
    main()


    """
    Example usage:
    python test_discriminator.py -o "experiments/no-noise adam-eps-1e-4/options-and-config.pickle" -c "experiments/no-noise adam-eps-1e-4/checkpoints/no-noise--epoch-200.pyt" -t "test_encoded_img.png"
    """
