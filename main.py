import os
import gc
import torch
import time
import cv2
import yaml
import random
import numpy as np
import torch
from datetime import datetime

from prompt_processing import encode_prompt, encode_prompt_sd
from mesh import load_mesh, update_mesh
from renderer import render
from light import sample_light_dir

from train import Trainer

import argparse
from omegaconf import OmegaConf
from torch.nn import Parameter, ParameterDict
import torchvision.transforms as transforms
import torch.nn.functional as F
from mesh import copy_mesh, write_obj


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--folder_name', type=str, default=None)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--prompt_cache_dir', type=str, default='cached_prompts')
    parser.add_argument('--guidance', type=str, default='if')
    parser.add_argument('--objaverse_eval', action='store_true')

    # data
    parser.add_argument('--mesh_location', type=str, default='data/model_cow.obj')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--use_directional_embeddings', action='store_true')

    parser.add_argument('--diffuse_texture_path', type=str, default=None)
    parser.add_argument('--no_diffuse_part', action='store_true')
    parser.add_argument('--no_diffuse_learning', action='store_true')

    parser.add_argument('--normals_texture_path', type=str, default=None)
    parser.add_argument('--no_normals_part', action='store_true')
    parser.add_argument('--no_normals_learning', action='store_true')

    parser.add_argument('--metallic_texture_path', type=str, default=None)
    parser.add_argument('--no_metallic_part', action='store_true')
    parser.add_argument('--no_metallic_learning', action='store_true')
    
    parser.add_argument('--roughness_texture_path', type=str, default=None)
    parser.add_argument('--no_roughness_part', action='store_true')
    parser.add_argument('--no_roughness_learning', action='store_true')
    
    parser.add_argument('--texture_dim', type=int, default=1024)
    parser.add_argument('--env_light_1', type=str, default='assets/kloofendal_28d_misty_puresky_2k.hdr')
    parser.add_argument('--scale_light_1', type=float, default=2.)
    parser.add_argument('--env_light_2', type=str, default='assets/kloofendal_28d_misty_puresky_2k.hdr')
    parser.add_argument('--scale_light_2', type=float, default=2.)
    parser.add_argument('--random_light_scale', action='store_true')

    # trainer
    parser.add_argument('--model_i', type=str, default="DeepFloyd/IF-I-XL-v1.0")
    parser.add_argument('--model_ii', type=str, default="DeepFloyd/IF-II-L-v1.0")
    parser.add_argument('--n_steps_1', type=int, default=500)
    parser.add_argument('--n_steps_2', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_time_schedule_1', action='store_true')
    parser.add_argument('--use_time_schedule_2', action='store_true')
    parser.add_argument('--lowres_noise_level', type=float, default=0.5)
    parser.add_argument('--use_gs_schedule_1', action='store_true')
    parser.add_argument('--guidance_scale_1', type=float, default=15.0)
    parser.add_argument('--use_gs_schedule_2', action='store_true')
    parser.add_argument('--guidance_scale_2', type=float, default=10.0)
    parser.add_argument('--min_noise_level_1', type=int, default=200) # in HiFA 300
    parser.add_argument('--max_noise_level_1', type=int, default=900)
    parser.add_argument('--min_noise_level_2', type=int, default=200) # in HiFA 300
    parser.add_argument('--max_noise_level_2', type=int, default=900)
    parser.add_argument('--no_random_background', action='store_true')
    parser.add_argument('--eval_plot_iter', type=int, default=100)
    parser.add_argument('--eval_renders_count', type=int, default=4)
    parser.add_argument('--gs_limit', type=int, default=None)
    parser.add_argument('--gamma_correction_1', action='store_true')
    parser.add_argument('--gamma_value_1', type=float, default=2.2)
    parser.add_argument('--gamma_correction_2', action='store_true')
    parser.add_argument('--gamma_value_2', type=float, default=2.2)
    parser.add_argument('--no_x0_clipping', action='store_true')
    parser.add_argument('--no_mesh_rotation', action='store_true')

    # optimizer
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lr_1', type=float, default=0.01)
    parser.add_argument('--lr_2', type=float, default=0.01)
    parser.add_argument('--gamma_scheduler', type=float, default=0.9)

    args = parser.parse_args()
    return args

def main():
    torch.cuda.reset_max_memory_allocated()
    start_time = time.time()
    args = parse_args()

    seed_all(args.seed)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # prompt = f"A DSLR photo of {args.prompt}, best quality, high quality, extremely detailed, good geometry"
    prompt = args.prompt
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    experiment_name = f"{args.prompt}_{timestamp}"
    
    if args.objaverse_eval:
        parts = args.prompt.split("_")
        name = " ".join(parts[:-1])
        number = parts[-1]
        experiment_name = f"{name.replace(' ', '_')}_{number}"
        # prompt = f"A DSLR photo of {name}, best quality, high quality, extremely detailed, good geometry"
        prompt = name
        args.prompt = name

    if args.folder_name == None:
        experiment_path = os.path.join(args.log_dir, experiment_name)
    else:
        experiment_path = os.path.join(args.log_dir, args.folder_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    print("Getting prompt embeddings...")
    if args.guidance == 'if':
        prompt_embeddings = encode_prompt(prompt, directional=args.use_directional_embeddings)
    elif args.guidance == 'sd':
        prompt_embeddings = encode_prompt_sd(prompt)

    torch.cuda.empty_cache()

    print("Initializing texture and starting training...")
    texture_dim = args.texture_dim
    use_predefined_texture = False
    
    diffuse_texture = torch.full((texture_dim, texture_dim, 3), 0.5)
    normals_texture = torch.zeros((texture_dim, texture_dim, 3))
    normals_texture[..., 2:3] = 1.
    # УБРАТЬ
    metallic_texture = torch.full((texture_dim, texture_dim, 1), 0.95)
    # metallic_texture = torch.full((texture_dim, texture_dim, 1), 0.5)
    roughness_texture = torch.full((texture_dim, texture_dim, 1), 0.5)

    # Load predefined textures if provided
    if args.diffuse_texture_path:
        use_predefined_texture = True
        diffuse_texture = cv2.resize(
            cv2.cvtColor(cv2.imread(args.diffuse_texture_path), cv2.COLOR_RGB2BGR), 
            (texture_dim, texture_dim)
        )
        diffuse_texture = transforms.ToTensor()(diffuse_texture).permute(1, 2, 0).contiguous()

    if args.normals_texture_path:
        use_predefined_texture = True
        normals_texture = cv2.resize(
            cv2.cvtColor(cv2.imread(args.normals_texture_path), cv2.COLOR_RGB2BGR), 
            (texture_dim, texture_dim)
        )
        normals_texture = transforms.ToTensor()(normals_texture).permute(1, 2, 0).contiguous()
    
    if args.metallic_texture_path:
        use_predefined_texture = True
        metallic_texture = cv2.resize(
            cv2.imread(args.metallic_texture_path, cv2.IMREAD_GRAYSCALE), 
            (texture_dim, texture_dim)
        )
        metallic_texture = transforms.ToTensor()(metallic_texture).permute(1, 2, 0).contiguous()

    if args.roughness_texture_path:
        use_predefined_texture = True
        roughness_texture = cv2.resize(
            cv2.imread(args.roughness_texture_path, cv2.IMREAD_GRAYSCALE), 
            (texture_dim, texture_dim)
        )
        roughness_texture = transforms.ToTensor()(roughness_texture).permute(1, 2, 0).contiguous()

    # Initialize material parameters
    # pbr_material_parameters = ParameterDict({
    #     'diffuse_texture' : Parameter(diffuse_texture.cuda()) if not args.no_diffuse else None,
    #     'normals_texture' : Parameter(normals_texture.cuda()) if not args.no_normals else None,
    #     'metallic_texture': Parameter(metallic_texture.cuda()) if not args.no_metallic else None,
    #     'roughness_texture': Parameter(roughness_texture.cuda()) if not args.no_roughness else None,
    # })
    pbr_material_parameters = ParameterDict()
    if not args.no_diffuse_part:
        pbr_material_parameters['diffuse_texture'] = Parameter(diffuse_texture.cuda())
    if not args.no_normals_part:
        pbr_material_parameters['normals_texture'] = Parameter(normals_texture.cuda())
    if not args.no_metallic_part:
        pbr_material_parameters['metallic_texture'] = Parameter(metallic_texture.cuda())
    if not args.no_roughness_part:
        pbr_material_parameters['roughness_texture'] = Parameter(roughness_texture.cuda())    
    if args.no_diffuse_learning:
        pbr_material_parameters['diffuse_texture'].requires_grad_(False)
    if args.no_normals_learning:
        pbr_material_parameters['normals_texture'].requires_grad_(False) 
    if args.no_metallic_learning:
        pbr_material_parameters['metallic_texture'].requires_grad_(False) 
    if args.no_roughness_learning:
        pbr_material_parameters['roughness_texture'].requires_grad_(False) 
    
    # Load mesh
    print("Loading mesh data...")
    my_mesh = load_mesh(
        args.mesh_location, 
        pbr_material_parameters,
        use_predefined_texture=use_predefined_texture
    )
    texture = pbr_material_parameters

    # Training process
    if args.guidance == 'sd':
        stage = 'i_sd'
        trainer = Trainer(my_mesh, texture, prompt_embeddings, stage, experiment_path, args)
        trainer.run_training_loop()
    else:
        # Phase 1
        start_time1 = time.time()
        if args.n_steps_1 > 0:
            stage = 'i'
            trainer = Trainer(my_mesh, texture, prompt_embeddings, stage, experiment_path, args)
            trainer.make_video(0)    
            trainer.run_training_loop()
            if not args.objaverse_eval:
                torch.save(texture, os.path.join(experiment_path, 'model_stage_i/texture_stage_i.pt')) 

            # Clean up phase 1
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
        
        phase1_time = time.time() - start_time1

        # Phase 2
        start_time2 = time.time()
        if args.n_steps_2 > 0:
            stage = 'ii'
            trainer = Trainer(my_mesh, texture, prompt_embeddings, stage, experiment_path, args)
            trainer.run_training_loop()
            if not args.objaverse_eval:
                torch.save(texture, os.path.join(experiment_path, 'model_stage_ii/texture_stage_ii.pt'))
        
            # Clean up phase 2
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

        phase2_time = time.time() - start_time2
        total_time = time.time() - start_time

    # Save experiment configuration
        args_dict = vars(args)
        args_dict['timestamp'] = timestamp
        args_dict['phase_1_time'] = phase1_time
        args_dict['phase_2_time'] = phase2_time
        args_dict['total_time'] = total_time
        config_path = os.path.join(experiment_path, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(args_dict, f, default_flow_style=False) 

        if args.n_steps_1 == 0 and args.n_steps_2 == 0:
            model_dir = os.path.join(experiment_path, f'model_stage_ii')
            os.mkdir(model_dir)
            write_obj(model_dir, my_mesh, texture)

        max_memory = torch.cuda.max_memory_allocated()
        print(f"\n\nExperiment completed. Results saved in {experiment_path}. Max memory used {max_memory / (1024 ** 3):.2f} GB")
    
    print(f"\n\nExperiment completed. Results saved in {experiment_path}.")

if __name__ == "__main__":
    main()