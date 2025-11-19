import torch

# def sample_light_dir(batch_size, device):
#     phi = torch.pi - 0.5 * torch.pi * (torch.rand(batch_size) - 0.5)
#     theta = 0.25 * torch.pi * torch.rand(batch_size)
#     light_dir = torch.stack([theta.cos() * phi.sin(),
#                              theta.sin(),
#                              theta.cos() * phi.cos()], -1)
#     return light_dir.to(device)

def sample_light_dir(batch_size, device, a=torch.pi, b=0.5*torch.pi):
    # phi = torch.pi * torch.rand(batch_size) - 0.5 * torch.pi
    phi =  a * torch.rand(batch_size) - b
    theta = 2 * torch.pi * torch.rand(batch_size)
    light_dir = torch.stack([theta.cos() * phi.sin(),
                             theta.sin(),
                             theta.cos() * phi.cos()], -1)
    return light_dir.to(device)

def sample_val_light_dir(device):
    phi = torch.tensor(-torch.pi)
    theta = torch.tensor(0.125 * torch.pi)
    light_dir = torch.stack([theta.cos() * phi.sin(),
                             theta.sin(),
                             theta.cos() * phi.cos()], -1)
    return light_dir.to(device)

def circular_light_motion(num_frames=10, device='cpu', radius=1.0, speed=1.0):
    phi = torch.linspace(0, 2 * torch.pi, num_frames) * speed
    theta = torch.tensor(0.25 * torch.pi)
    light_dir_seq = torch.stack([radius * torch.cos(theta) * torch.ones_like(phi) * torch.sin(phi),
                                 radius * torch.sin(theta) * torch.ones_like(phi),
                                 radius * torch.cos(theta) * torch.ones_like(phi) * torch.cos(phi)], -1)
    return light_dir_seq.to(device)