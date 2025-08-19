import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from zarth_utils.config import Config


def main():
    # ----------------------------
    # Configuration Parameters
    # ----------------------------
    config = Config(
        default_config_dict={
            "data_dir": "./imagenet/val",
            "batch_size": 256,
            "num_workers": 4,
            "device": "cuda",
            "model_weight_name": "AlexNet_Weights.IMAGENET1K_V1",
            "output_dir": "./raw/imagenet_predictions",
        },
        use_argparse=True,
    )
    config.show()
    os.makedirs(config.output_dir, exist_ok=True)
    path_save = os.path.join(
        config.output_dir, config.model_weight_name + "_predictions.pth"
    )
    if os.path.exists(path_save):
        print(f"Predictions already exist at '{path_save}'. Exiting...")
        return

    # ----------------------------
    # Define Data Transformations
    # ----------------------------
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalization parameters specific to the ImageNet dataset
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ----------------------------
    # Load the ImageNet Validation Dataset
    # ----------------------------
    print("Loading the ImageNet validation dataset...")
    val_dataset = datasets.ImageFolder(config.data_dir, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # ----------------------------
    # Load the Pre-trained Model from Torch Hub
    # ----------------------------
    model_name = "_".join(
        config.model_weight_name.split(".")[0].split("_")[:-1]
    ).lower()
    weight_name = config.model_weight_name.split(".")[1]
    print(
        f"Loading the pre-trained model '{model_name}' with weights '{weight_name}' from Torch Hub..."
    )
    model = torch.hub.load("pytorch/vision:v0.20.0", model_name, weights=weight_name)
    model.eval()
    model.to(config.device)

    # ----------------------------
    # Generate Predictions
    # ----------------------------
    print("Generating predictions...")
    all_predictions = []
    all_image_paths = []
    all_labels = []
    all_accs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(config.device)

            # Forward pass
            outputs = model(images)

            # Apply softmax to get probabilities (optional)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Move predictions back to CPU
            probabilities = probabilities.cpu()

            # Store predictions and image paths
            all_predictions.append(probabilities)
            batch_image_paths = [
                path
                for path, _ in val_loader.dataset.samples[
                    batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size
                ]
            ]
            all_image_paths.extend(batch_image_paths)
            all_labels.extend(labels)
            all_accs.extend(probabilities.argmax(dim=-1) == labels)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    print("Accuracy on the validation set: ", sum(all_accs) / len(all_accs))

    # ----------------------------
    # Save Predictions to a File
    # ----------------------------
    print(f"Saving predictions to '{path_save}'...")
    torch.save(
        {
            "predictions": all_predictions,
            "image_paths": all_image_paths,
            "labels": all_labels,
            "accs": all_accs,
        },
        path_save,
    )

    print("Finished generating and saving predictions.")


if __name__ == "__main__":
    main()
