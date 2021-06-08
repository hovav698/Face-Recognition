import torch

class SiameseNetwork(torch.nn.Module):
    def __init__(self, feature_dim):
        super(SiameseNetwork, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(13 * 18 * 64, 128),  # 60>29>13,80>39>18
            torch.nn.ReLU(),
            torch.nn.Linear(128, feature_dim),
        )

    # the model output is the distance between the embedding of the images
    def forward(self, img1, img2):
        emedding1 = self.model(img1)
        emedding2 = self.model(img2)
        return torch.norm(emedding1 - emedding2, dim=-1)

