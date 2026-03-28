import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
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


def save_image(tensor, filename):
    """Save a tensor as image file"""
    # Convert from CHW to HWC and from [-1, 1] to [0, 255]
    img = (tensor.permute(1, 2, 0).numpy() + 1) / 2 * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(filename)


def save_images_fixed(original_images, watermarked_images, name, folder):
    """Custom save images function without using torchvision"""
    # scale values to range [0, 1] from original range of [-1, 1]
    images = (original_images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2
    
    # Save first image from batch
    if images.shape[0] > 0:
        original_filename = os.path.join(folder, '{}_original_img.png'.format(name))
        img_np = images[0].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(original_filename)
        
        watermarked_filename = os.path.join(folder, '{}_encoded_img.png'.format(name))
        wm_np = watermarked_images[0].permute(1, 2, 0).numpy()
        wm_np = np.clip(wm_np * 255, 0, 255).astype(np.uint8)
        Image.fromarray(wm_np).save(watermarked_filename)
        
        print('Saved images:')
        print('  - Original: {}'.format(original_filename))
        print('  - Encoded:  {}'.format(watermarked_filename))


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def parse_binary_message(message_str, expected_length):
    """Parse binary string message (e.g., '101010...')"""
    # Remove any whitespace
    message_str = message_str.replace(' ', '')
    
    # Validate input
    if not all(c in '01' for c in message_str):
        raise ValueError("Message must contain only '0' and '1' characters")
    
    # Auto-adjust message length
    if len(message_str) > expected_length:
        # Truncate if too long
        print('Warning: Message too long ({} bits), truncating to first {} bits'.format(len(message_str), expected_length))
        message_str = message_str[:expected_length]
    elif len(message_str) < expected_length:
        # Pad with zeros if too short
        print('Warning: Message too short ({} bits), padding with {} zeros to reach {} bits'.format(len(message_str), expected_length - len(message_str), expected_length))
        message_str = message_str + '0' * (expected_length - len(message_str))
    
    # Convert to numpy array
    return np.array([int(c) for c in message_str], dtype=np.float32)


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using device:', device)

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The image to watermark')
    parser.add_argument('--message', '-m', type=str,
                        help='Custom binary message (e.g., "101010..."). If longer than 30 bits, will be truncated; if shorter, will be padded with zeros. If not provided, random message will be generated.')

    args = parser.parse_args()

    print('Loading options from:', args.options_file)
    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    print('Message length required by model:', hidden_config.message_length)
    print('Image size: {}x{}'.format(hidden_config.H, hidden_config.W))
    
    noiser = Noiser(noise_config, device)

    print('Loading checkpoint from:', args.checkpoint_file)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    print('Model loaded successfully!')

    print('Loading image from:', args.source_image)
    image_pil = Image.open(args.source_image)
    image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    print('Cropped image shape:', image.shape)
    
    image_tensor = to_tensor(image).to(device)
    image_tensor.unsqueeze_(0)
    print('Image tensor shape:', image_tensor.shape)

    # Generate or parse message
    if args.message:
        print('\nUsing custom message...')
        message_np = parse_binary_message(args.message, hidden_config.message_length)
        message = torch.Tensor(message_np).unsqueeze(0).to(device)
    else:
        print('\nGenerating random message...')
        message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                        hidden_config.message_length))).to(device)

    print('Testing watermarking...')
    losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
    
    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    message_detached = message.detach().cpu().numpy()
    
    print('\n=== Results ===')
    print('Original message: {}'.format(message_detached.astype(int)))
    print('Decoded message:  {}'.format(decoded_rounded.astype(int)))
    bit_error_rate = np.mean(np.abs(decoded_rounded - message_detached))
    print('Bit error rate:  {:.3f}'.format(bit_error_rate))
    
    save_images_fixed(image_tensor.cpu(), encoded_images.cpu(), 'test', '.')


if __name__ == '__main__':
    main()

    
    """
    test_model.py -o "runs\train 2026.03.21--11-33-51\options-and-config.pickle" -c "runs\train 2026.03.21--11-33-51\checkpoints\train--epoch-300.pyt" -s "dataset1\val\val_class\000000553000.jpg" -m 110111
    """