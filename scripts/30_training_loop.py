import os, torch, numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from vit_pytorch import ViT, Dino
from pathlib import Path

HOME=str(Path.home())
    
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data[idx])
        img = read_image(img_path)
        return img / 255 

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") # if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 200,'shuffle': True, 'num_workers': 2}

custom_dataset = CustomDataset(data_dir=f'{HOME}/workspace/hack_team_01/data/processed/patch_256')
data_loader = DataLoader(custom_dataset, **params)

model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,                # i dont think this matters in self supervised formulation
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

learner = Dino(
    model,
    image_size = 256,
    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
).to(device)

opt = torch.optim.Adam(learner.parameters(), lr = 3e-4)

print(torch.cuda.is_available() , flush=True )

torch.cuda.cudart().cudaProfilerStart()
for epoch in range(2):
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        # forward
        torch.cuda.nvtx.range_push("iteration{}".format(i*(epoch+1)))
        loss = learner(batch)

        # backward 
        torch.cuda.nvtx.range_push("backward")
        opt.zero_grad()
        loss.backward()

        # optimizer
        torch.cuda.nvtx.range_push("opt.step()")
        opt.step()
        learner.update_moving_average() 


        if i % 20 == 0:
            print(f"Step {i} :  {torch.cuda.memory_allocated(device=device)/(1024**3) } [GB]", flush=True)
    torch.cuda.empty_cache()

torch.cuda.cudart().cudaProfilerStop()

#torch.save(model.state_dict(), './workspace/checkpoints/pretrained-net_modis_256_256_patch32_mod.pt')
os.makedirs(f"{HOME}/run/profile",exist_ok=True)
torch.save(model.state_dict(), f'{HOME}/run/profile/pretrained-net_modis_256_256_patch32_mod.pt')

