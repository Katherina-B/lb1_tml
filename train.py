import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract the directory path from the log file path
log_dir = os.path.dirname(config["logging"]["file"])

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=config["logging"]["level"],
    format=config["logging"]["format"],
    handlers=[
        logging.FileHandler(config["logging"]["file"]),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Type hints
Dataset = Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]
ModelOutput = Tuple[nn.Module, optim.Optimizer, nn.CrossEntropyLoss]


def load_data(data_dir: str) -> Dataset:
    """Load and preprocess the dataset.

    Args:
        data_dir (str): The directory where the dataset is stored.

    Returns:
        Dataset: A tuple containing the train, validation, and test datasets.
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the train dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)

    # Load the test dataset
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)

    # Calculate the validation split size
    total_size = len(train_dataset)
    val_size = int(config["dataset"]["val_split"] * total_size)
    train_size = total_size - val_size

    # Split the train dataset into train and validation sets
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["dataset"]["random_seed"])
    )

    return train_dataset, val_dataset, test_dataset


def create_model() -> ModelOutput:
    """Create the model, optimizer, and loss function."""
    # Create the model
    model = getattr(torchvision.models, config["model"]["name"])(
        pretrained=config["model"]["pretrained"],
        num_classes=config["model"]["num_classes"],
    )

    # Freeze base layers if specified
    if config["model"]["freeze_base"]:
        for param in model.parameters():
            param.requires_grad = False

    # Create the optimizer
    optimizer_config = config["training"]["optimizer"]
    optimizer = getattr(optim, optimizer_config["name"])(
        model.parameters(), lr=optimizer_config["lr"]
    )

    # Create the loss function
    loss_fn = getattr(nn, config["training"]["loss"] + "Loss")()

    return model, optimizer, loss_fn


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Train the model and evaluate on the validation set.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer for the model.
        loss_fn (nn.CrossEntropyLoss): The loss function.
        train_loader (DataLoader): The data loader for the training set.
        val_loader (DataLoader): The data loader for the validation set.
        device (str, optional): The device to use for training (CPU or GPU). Defaults to "cuda" if available, else "cpu".

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics on the validation set.
    """
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        train_accuracy = 100.0 * train_correct / train_total
        val_accuracy = 100.0 * val_correct / val_total

        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/train_total:.4f}, Train Accuracy={train_accuracy:.2f}%")
        logger.info(f"Epoch {epoch+1}: Val Loss={val_loss/val_total:.4f}, Val Accuracy={val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if config["artifacts"]["save_best_model"]:
                torch.save(model.state_dict(), os.path.join(config["artifacts"]["output_dir"], "best_model.pth"))

    return {"accuracy": best_accuracy / 100.0}


def main() -> None:
    # Load and preprocess the dataset
    train_dataset, val_dataset, test_dataset = load_data(config["data"]["local_dir"])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    # Create the model, optimizer, and loss function
    model, optimizer, loss_fn = create_model()

    # Train the model
    metrics = train(model, optimizer, loss_fn, train_loader, val_loader)
    logger.info(f"Validation metrics: {metrics}")

    # Save artifacts
    os.makedirs(config["artifacts"]["output_dir"], exist_ok=True)
    if config["artifacts"]["save_best_model"]:
        torch.save(model.state_dict(), os.path.join(config["artifacts"]["output_dir"], "best_model.pth"))
    if config["artifacts"]["save_logs"]:
        # Save logs
        with open(os.path.join(config["artifacts"]["output_dir"], "training.log"), "w") as log_file:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file.write(handler.stream.getvalue())
    if config["artifacts"]["save_metrics"]:
        # Save metrics
        with open(os.path.join(config["artifacts"]["output_dir"], "metrics.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file)


if __name__ == "__main__":
    main() 
