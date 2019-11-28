from dataLoader import DirtyDocumentsDataset_Test
from torch.utils.data import DataLoader
    
testdataset = DirtyDocumentsDataset_Test()
test_loader = DataLoader(testdataset, batch_size=1, shuffle=False)

weight_savename = 'd:/dataset/weights/DAE/deno_autoencoder.pth'
try:
    encoder, decoder = torch.load(weight_savename)
    print("\n--------model restored--------\n")
except:
    pass

#######################################
#               test                  #
#######################################

for img in test_loader:
    img = img.to(device, dtype=torch.float)
    with torch.no_grad():
        output = encoder(img)
        output = decoder(output)
        loss = mse_loss(output, img)
        
        output = chw2hwc(output.numpy()[0,:])
        img = chw2hwc(img.numpy()[0,:])
        print(np.unique(output))
        print(output.shape)
        print(img.shape)
        
        plt.imshow(output)
        plt.show()
        
        plt.imshow(img)
        plt.show()
        
        print("loss:", loss)
        break