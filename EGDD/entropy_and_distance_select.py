import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import shutil
from scipy.stats import entropy
from util.pos_embed import interpolate_pos_embed
import models_vit

def extract_features(model, image):
    model.eval()
    with torch.no_grad():
        features = model.forward_features(image)
    return features

def get_args_parser():
    parser = argparse.ArgumentParser('MAE inference for image classification in places365', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # * Finetuning params
    parser.add_argument('--finetune', default='./',
                        help='finetune from places365 data set')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)

    # Dataset parameters
    parser.add_argument('--error_images_path', default='./places365_12_ft_12_classes_all_error', type=str,
                        help='dataset path')
    parser.add_argument('--image_data_path', default='./generated_with_concept_3000', type=str,
                        help='dataset path')
    parser.add_argument('--output_path', default='./entropy_distance_based_image_select_data', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=12, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser

def main(args):
    device = torch.device(args.device)
    # Load the fine-tuned model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    checkpoint = torch.load(args.finetune, map_location='cpu')
    checkpoint_model = checkpoint['model']

    # Interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # Load the fine-tuned model
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    # Define transforms for images
    transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * args.input_size), interpolation=Image.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Adjust according to your model
    ])

    # Extract features for each class from error images
    class_features = {}
    for class_folder in os.listdir(args.error_images_path):
        class_path = os.path.join(args.error_images_path, class_folder)
        class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        class_embeddings = []
        for img_path in tqdm(class_images, desc=f'Extracting features for class {class_folder}'):
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            features = extract_features(model, image)
            class_embeddings.append(features.cpu().numpy())

        class_embeddings = torch.tensor(class_embeddings).mean(dim=0).squeeze().numpy()
        class_features[class_folder] = class_embeddings

    for class_name in os.listdir(args.image_data_path):
        class_dir = os.path.join(args.image_data_path, class_name)

        if not os.path.isdir(class_dir):
            continue

        print(f"Processing images in class: {class_name}")

        image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if
                       img.endswith(('.jpg', '.png', '.jpeg'))]

        # Calculate entropy and distance for each image
        entropies = []
        distances = []
        for image_path in tqdm(image_paths, desc="Calculating entropy and distance", unit="image"):
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Get model prediction
            with torch.no_grad():
                model.eval()
                output = model(img_tensor)

            # Calculate softmax probabilities and entropy
            softmax_probs = F.softmax(output, dim=1).cpu().numpy()[0]
            entropies.append(entropy(softmax_probs))

            # Calculate distance to class feature
            features = extract_features(model, img_tensor).cpu().numpy()
            class_feature = class_features[class_name]
            distance = ((features - class_feature) ** 2).sum()
            distances.append(distance)

        # Normalize entropies and distances
        normalized_entropies = [(e - min(entropies)) / (max(entropies) - min(entropies)) for e in entropies]
        normalized_distances = [(d - min(distances)) / (max(distances) - min(distances)) for d in distances]

        # Combine normalized entropies and distances with weights
        combined_scores = [0.5 * e + 0.5 * d for e, d in zip(normalized_entropies, normalized_distances)]

        # Select top 1000 images based on combined scores
        selected_indices = sorted(range(len(combined_scores)), key=lambda k: combined_scores[k], reverse=True)[:1000]

        output_class_dir = os.path.join(args.output_path, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for idx in tqdm(selected_indices, desc="Copying images", unit="image"):
            selected_image_path = image_paths[idx]
            shutil.copy2(selected_image_path, output_class_dir)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
