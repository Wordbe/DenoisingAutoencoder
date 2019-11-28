class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(C, 32, 3, padding=1),   # batch x 32 x H x W
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 32, 3, padding=1),   # batch x 32 x H x W
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x H x W
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x H x W
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2)   # batch x 64 x H/2 x W/2
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x H/2 x W/2
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, 3, padding=1),  # batch x 128 x H/2 x W/2
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x H/4 x W/4
                        nn.BatchNorm2d(256),
                        nn.ReLU()
        )
        
                
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(BATCH_SIZE, -1)
        return out

# Transpose2d, upsampling 방법 변경?

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), # batch x 128 x H/4 x W/4
                        nn.BatchNorm2d(128),            
                        nn.ReLU(),
                        nn.ConvTranspose2d(128, 128, 3, 1, 1),   # batch x 128 x H/2 x W/2
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128, 64, 3, 1, 1),    # batch x 64 x H/2 x W/2
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 64, 3, 1, 1),     # batch x 64 x H/2 x W/2
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 3, 1, 1),     # batch x 32 x H/2 x W/2
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, 32, 3, 1, 1),     # batch x 32 x H/2 x W/2
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, C, 3, 2, 1, 1),    # batch x C x H x W
                        nn.ReLU()
        )
        
    def forward(self, x):
        out = x.view(-1, 256, H//4, W//4)
        out = self.layer1(out)
        out = self.layer2(out)
        return out