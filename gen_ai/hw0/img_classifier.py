import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse
import wandb

img_size = (256,256)
num_labels = 3
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
denormalize = T.Normalize(
    mean=[-m/s for m, s in zip(normalize_mean, normalize_std)],
    std=[1/s for s in normalize_std])

categories = ['parrot', 'narwhal', 'axolotl']

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        image = Image.open(img_name).convert("RGB")  # Assuming RGB images
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(args):
    transform_img = T.Compose([
        T.ToTensor(),
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        T.CenterCrop(img_size),  # Center crop to 256x256
        T.Normalize(mean=normalize_mean, std=normalize_std), # Normalize each color dimension
        ])

    if args.grayscale:
        # Append grayscale transformation
        transform_img.transforms.extend([T.Grayscale(num_output_channels=1)])

    train_data = CsvImageDataset(
        csv_file='./data/img_train.csv',
        transform=transform_img,
    )
    test_data = CsvImageDataset(
        csv_file='./data/img_test.csv',
        transform=transform_img,
    )
    val_data = CsvImageDataset(
        csv_file='./data/img_val.csv',
        transform=transform_img,
    )

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader, val_dataloader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # First layer input size must be the dimension of the image

        if args.grayscale:
            input_size = img_size[0] * img_size[1] * 1
        else:
            input_size = img_size[0] * img_size[1] * 3

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

        total_params = sum(
            param.numel() for param in self.parameters()
        )

        print(f"Total number of model parameters: {total_params}")

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class NeuralNetworkCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.forward_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=4),
            nn.LayerNorm([128, 64, 64]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, padding=3),
            nn.LayerNorm([128, 64, 64]),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=(128 * 32 * 32), out_features=3),
        )

        total_params = sum(
            param.numel() for param in self.parameters()
        )

        print(f"Total number of model parameters: {total_params}")

    def forward(self, x):
        return self.forward_stack(x)

def train_one_epoch(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        """ temp = X
        for i, layer in enumerate(model.children()):
            temp = layer(temp)
            print(f"Layer {i} ({layer.__class__.__name__}): Output Shape = {temp.shape}") """

        pred = model(X)
        loss = loss_fn(pred, y)
        batch_size = len(y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item() / batch_size
        current = (batch + 1) * batch_size

        if args.use_wandb:
            wandb.log({"Train Batch Loss": loss, "Num example": current + t * size})

        if batch % 10 == 0:
            print(f"Train batch avg loss = {loss:>7f}  [{current:>5d}/{size:>5d}], Num example: {current}")

def evaluate(dataloader, dataname, model, loss_fn, t):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            if t % 5 == 0 and args.log_images and batch == 0 and args.use_wandb:
                # Log an example image (after denormalizing) to wandb
                for i in range(len(y)):
                    Xi_denorm = denormalize(X[i])
                    predicted_label = categories[pred[i].argmax().item()]
                    real_label = categories[y[i].item()]
                    caption = f"{predicted_label}/{real_label}"
                    wandb.log({"image": wandb.Image(Xi_denorm, caption=caption, mode="RGB") })

            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_loss /= size
    correct /= size
    accuracy = 100 * correct
    print(f"{dataname} accuracy = {accuracy:>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")

    return accuracy, avg_loss

def main(args):
    torch.manual_seed(10999)

    if args.use_wandb:
        wandb.login()
        wandb.init(
            project="img_classifier",
            config={
                "epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "model": args.model,
                "grayscale": args.grayscale,
            })
    else:
        wandb.init(mode='disabled')

    print(f"Using {device} device")
    train_dataloader, test_dataloader, val_dataloader = get_data(args)

    if args.model == 'simple':
        model = NeuralNetwork().to(device)
    elif args.model == 'cnn':
        model = NeuralNetworkCNN().to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    print(model)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for t in range(args.n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, t)
        (train_accuracy, train_loss) = evaluate(train_dataloader, "Train", model, loss_fn, t)
        (test_accuracy, test_loss) = evaluate(test_dataloader, "Test", model, loss_fn, t)
        (val_accuracy, val_loss) = evaluate(val_dataloader, "Validation", model, loss_fn, t)
        wandb.log({
            "Train Accuracy": train_accuracy,
            "Train Loss": train_loss,
            "Test Accuracy": test_accuracy,
            "Test Loss": test_loss,
            "Validation Accuracy": val_accuracy,
            "Validation Loss": val_loss,
            "epoch": t,
        })

    print("Done!")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Load the model (just for the sake of example)
    """ model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True)) """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use Weights and Biases for logging')
    parser.add_argument('--n_epochs', type=int, default=5, help='The number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='The learning rate for the optimizer')
    parser.add_argument('--model', type=str, choices=['simple', 'cnn'], default='simple', help='The model type')
    parser.add_argument('--grayscale', action='store_true', default=False, help='Use grayscale images instead of RGB')
    parser.add_argument('--log_images', action='store_true', default=False, help='Log images to wandb')

    args = parser.parse_args()

    main(args)