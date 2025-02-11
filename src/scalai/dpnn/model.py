"""Model definition for AlexNet."""

import torch

# MODEL
# Define neural network by subclassing PyTorch's nn.Module.
# Save to a separate Python module file `model.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


class AlexNet(torch.nn.Module):
    """AlexNet NN architecture.

    Attributes
    ----------
    features : torch.nn.container.Sequential
        The convolutional feature-extractor part.
    avgpool : torch.nn.AdaptiveAvgPool2d
        An adaptive pooling layer to handle different input sizes.
    classifier : torch.nn.container.Sequential
        The fully connected linear part.

    Methods
    -------
    __init__()
        The constructor defining the network's architecture.
    forward()
        The forward pass.
    """

    # Initialize neural network layers in __init__.
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        """Initialize AlexNet architecture.

        Parameters
        ----------
        num_classes : int
            The number of classes in the underlying classification problem.
        dropout : float
            The dropout probability.
        """
        super().__init__()
        self.features = torch.nn.Sequential(
            # AlexNet has 8 layers: 5 convolutional layers, some followed by max-pooling (see figure),
            # and 3 fully connected layers. In this model, we use nn.ReLU between our layers.
            # nn.Sequential is an ordered container of modules.
            # The data is passed through all the modules in the same order as defined.
            # You can use sequential containers to put together a quick network.
            #
            # IMPLEMENT FEATURE-EXTRACTOR PART OF ALEXNET HERE!
            # 1st convolutional layer (+ max-pooling)
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            ## 2nd convolutional layer (+ max-pooling)
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            ## 3rd + 4th convolutional layer
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            ## 5th convolutional layer (+ max-pooling)
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Average pooling to downscale possibly larger input images.
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            # IMPLEMENT FULLY CONNECTED PART HERE!
            # 6th, 7th + 8th fully connected layer
            # The linear layer is a module that applies a linear transformation
            # on the input using its stored weights and biases.
            # 6th fully connected layer (+ dropout)
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            ## 7th fully connected layer (+ dropout)
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=dropout),
            # 8th (output) layer
            torch.nn.Linear(4096, num_classes),
        )

    # Forward pass: Implement operations on the input data, i.e., apply model to input x.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The model's output.
        """
        # IMPLEMENT OPERATIONS ON INPUT DATA x HERE!
        ## Apply feature-extractor part to input.
        x = self.features(x)
        ## Apply average-pooling part.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten.
        ## Apply fully connected part.
        x = self.classifier(x)
        return x
