import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import wandb
import os

wandb.login()
run = wandb.init(
    project="mldevops_exercise",
    config={
        "learning_rate": 1e-3,
        "epochs": 30,
        "batch_size": 32,
    },
)

tensor_transform = transforms.ToTensor()
dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tensor_transform)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=wandb.config.batch_size, shuffle=True)


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = AE()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-8)

losses = []
for epoch in range(wandb.config.epochs):
    for image, _ in loader:
        image = image.reshape(-1, 28 * 28)

        reconstructed = model(image)

        loss = loss_function(reconstructed, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    wandb.log({"loss": loss.item(), "epoch": epoch})

    print(f"Epoch {epoch+1}/{wandb.config.epochs}, Loss: {loss.item()}")

plt.style.use("fivethirtyeight")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(losses[-100:])
plt.show()

with torch.no_grad():
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(image[i].reshape(28, 28), cmap="gray")
        plt.axis("off")

        print(f"Reconstructed image {i}: min={reconstructed[i].min()}, max={reconstructed[i].max()}")

        plt.subplot(2, 10, i + 11)
        plt.imshow(reconstructed[i].detach().numpy().reshape(28, 28), cmap="gray")
        plt.axis("off")

plt.show()

wandb.finish()
