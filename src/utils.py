import torch
import numpy as np
from torch.nn import Parameter, ParameterDict

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_texture(texture_dim):

    diffuse_texture = torch.full((texture_dim, texture_dim, 3), 0.5)
    normals_texture = torch.zeros((texture_dim, texture_dim, 3))
    normals_texture[..., 2] = 1.0
    metallic_texture = torch.full((texture_dim, texture_dim, 1), 0.5)
    roughness_texture = torch.full((texture_dim, texture_dim, 1), 0.5)
    
    pbr_material_parameters = ParameterDict({
        'diffuse_texture': Parameter(diffuse_texture.cuda()),
        'normals_texture': Parameter(normals_texture.cuda()),
        'metallic_texture': Parameter(metallic_texture.cuda()),
        'roughness_texture': Parameter(roughness_texture.cuda()),
    })

    return pbr_material_parameters

def save_texture(experiment_path, texture, stage):
    save_dir = os.path.join(experiment_path, f'model_stage_{stage}')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(texture, os.path.join(save_dir, f'texture_stage_{stage}.pt'))
    print(f"Stage {stage} textures saved to {save_dir}")