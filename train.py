from torch.utils.data import DataLoader
from dataLoader import DirtyDocumentsDataset
from model import Encoder, Decoder
import argparse

# Set Hyperparameters
N_EPOCH = 100
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
C, H, W = 1, 256, 540

# trainpath = 'd:/dataset/dirty_paper/train/train_cleaned'
# noisepath = 'd:/dataset/dirty_paper/train/train'
# testpath = 'd:/dataset/dirty_paper/test/test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu" # for test
print(device)

cropped_height = 256
cropped_width = 256
transform_train = transforms.Compose([
                                       RandomCrop((cropped_height, cropped_width)),
                                       Rescale((H, W)),
                                       ImgAugTransform(),
                                       ToTensor()
                                       ])

dirty_dir1 = 'd:/dataset/dirty_paper/train/train/'
dirty_dir2 = 'd:/dataset/dirty_paper/train_bright/train_bright/'
dirty_dir3 = 'd:/dataset/dirty_paper/train_crumpled/train_crumpled/'

clean_dir1 = 'd:/dataset/dirty_paper/train/train_cleaned/'
clean_dir2 = 'd:/dataset/dirty_paper/train_bright/train_bright_clean/'
clean_dir3 = 'd:/dataset/dirty_paper/train_crumpled/train_crumpled_clean/'

dirty_dirs = [dirty_dir1, dirty_dir2, dirty_dir3]
clean_dirs = [clean_dir1, clean_dir2, clean_dir3]

traindataset = DirtyDocumentsDataset(clean_dirs, dirty_dirs, transform_train)
train_loader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)


encoder = Encoder().to(device)
decoder = Decoder().to(device)

parameters = list(encoder.parameters()) + list(decoder.parameters())
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)

weight_savename = 'd:/dataset/weights/DAE/deno_autoencoder.pth'

# try:
#     encoder, decoder = torch.load(weight_savename)
#     print("\n--------model restored--------\n")
# except:
#     pass


########################################
#               train                  #
########################################

min_loss = float("inf")

for ith, epoch in enumerate(range(N_EPOCH), start=1):
    loss_per_epoch = 0.0
    
    for batch in tqdm(train_loader):
        img = batch['clean_img'].to(device, dtype=torch.float)
        noise_img = batch['dirty_img'].to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        output = encoder(noise_img)
        output = decoder(output)
        loss = mse_loss(output, img)
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss
        
        if loss < min_loss:
            torch.save([encoder, decoder], weight_savename)
            min_loss = loss
            
    loss_per_epoch /= len(train_loader)
    print('epoch: {}, train loss: {}'.format(epoch + 1, loss_per_epoch))
    
    # Log on the tensorboard
    writer.add_scalar('training loss',
                        loss_per_epoch,
                        epoch * ith)
    
if __name__ == '__main__':
    