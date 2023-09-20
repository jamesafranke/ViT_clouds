import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from vit_pytorch import ViT, Dino
from pathlib import Path
torch.backends.cuda.matmul.allow_tf32 = True

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

vit = ViT(image_size = 1024, patch_size = 32, num_classes = 1000, dim = 1024, depth = 6, heads = 8,  mlp_dim = 2048)

learner = Dino(
    vit,
    image_size = 1024,
    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temp, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder -anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers -anywhere from 0.9 to 0.999 was ok
)


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = learner.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    custom_dataset = CustomDataset(data_dir=f'{HOME}/workspace/hack_team_01/data/processed/patch_1024')
    data_loader = DataLoader(
        dataset=custom_dataset,
        batch_size=32,
        shuffle=False,
        sampler=DistributedSampler(custom_dataset),
    )

    for epoch in range(3):
        print(epoch)
        data_loader.sampler.set_epoch(epoch)

        for batch in data_loader:
            batch = batch.to(device_id)
            loss = ddp_model(batch)
            opt = torch.optim.Adam(learner.parameters(), lr = 3e-4)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() 

        dist.destroy_process_group()

if __name__ == "__main__":
    print(torch.cuda.device_count())
    demo_basic()
