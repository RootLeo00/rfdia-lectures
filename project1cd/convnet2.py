import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

class ConvNet2(nn.Module):
    def __init__(self, channels = 3):
        super(ConvNet2, self).__init__()

        self.channels = channels # to handle color images
        self.conv1 = 32
        self.conv2 = 64
        self.conv3 = 64
        self.fc4 = 1000
        self.fc5 = 10

        # conv net as feature extractor
        self.features = nn.Sequential(
            #--> input: 4x3x28x28 (color image)
            nn.Conv2d(self.channels, 32, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0), # --> output: [4, 32, 16, 16]

            nn.Conv2d(32, 64, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0), #--> ouput: [4, 64, 8, 8])

            nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0), #--> output: [4, 64, 4, 4])
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, self.fc4),
            nn.ReLU(),
            nn.Linear(self.fc4, self.fc5),

            # Reminder: The softmax is included in the loss, do not put it here
            # nn.Softmax()
        )


    def forward(self, input):
        bsize = input.size(0) # batch size
        output = self.features(input) # output of the conv layers

        output = output.view(bsize, -1) # we flatten the 2D feature maps into one 1D vector for each input
        output = self.classifier(output) # we compute the output of the fc layers

        return output
    
    def count_parameters(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params
    
