import os
import math
import torch
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

import envlight
from renderer import render
from guidance import SDSLoss
from camera import get_camera
from torch.nn import ParameterDict
from our_kaolin import rot33_rotate
from mesh import copy_mesh, write_obj
from prompt_processing import get_view_direction


class Trainer:
    def __init__(
        self, 
        mesh, 
        texture, 
        prompt_embeddings, 
        stage, 
        config,
        device='cuda:0'
    ):
        self.device = device
        self.mesh = mesh
        self.texture = texture
        self.prompt_embeddings = prompt_embeddings
        self.stage = stage
        
        self.batch_size = config.batch_size
        self.objaverse_eval = config.objaverse_eval
        self.eval_plot_iter = config.eval_plot_iter
        self.experiment_path = config.experiment_path
        self.use_directional_embeddings = config.use_directional_embeddings      

        self.camera_target, self.r = self._get_camera_parameters()
        
        self.dir2name = {
            0: 'front view',
            1: 'side view',
            2: 'backside view',
            3: 'top view',
            4: 'bottom view',
        }

        self._setup_stage(config)
        self._setup_validation_cameras(config)
        
        print(f'Initialized training for stage {stage}')

    
    def _setup_stage(self, config):

        if self.stage == 'i':
            self.num_training_steps = config.num_steps_i
            self.lr = config.lr_i
            self.guidance_scale = config.guidance_scale_i
            
            model_name = config.model_i
            print(f"Stage I using model: {model_name}")
            self.loss = SDSLoss(stage='i', model_name=model_name, device=self.device)
            
        elif self.stage == 'ii':
            self.num_training_steps = config.num_steps_ii
            self.lr = config.lr_ii
            self.guidance_scale = config.guidance_scale_ii
            self.lowres_noise_level = config.lowres_noise_level
            
            model_name = config.model_ii
            print(f"Stage II using model: {model_name}")
            self.loss = SDSLoss(stage='ii', model_name=model_name, device=self.device)
                        
            self.original_mesh = copy_mesh(self.mesh.detach(), self.texture)        
        
        self.light = envlight.EnvLight(
            config.env_light,
            scale=config.light_scale,
            device=self.device
        )
        
        self.optimizer = torch.optim.Adam(
            self.texture.parameters(),
            lr=self.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=config.gamma_scheduler
        )

    def _get_camera_parameters(self):
        camera_target = torch.tensor([0., 0., 0.])
        r = 2 * torch.tensor(2.2)
        self.min_radius_scale = 0.5
        self.max_radius_scale = 1.5
        return camera_target, r

    def _setup_validation_cameras(self, config):

        eval_renders_count = 4
        self.val_phis = torch.linspace(0, 2 * torch.pi, eval_renders_count + 1)[:-1].view(1, -1)
        self.val_thetas = torch.zeros(1, eval_renders_count)

        self.val_camera = []
        for i in range(0, len(self.val_phis[0]), 10):
            phi = self.val_phis[0][i:i + 10]
            theta = self.val_thetas[0][i:i + 10]
            
            val_eye = self.r * torch.stack([
                phi.cos() * theta.cos(),
                torch.ones_like(phi) * theta.sin(),
                phi.sin() * theta.cos()
            ], dim=-1).view(-1, 3)
            
            temp_val_camera, _, _ = get_camera(len(val_eye), self.camera_target, self.r, val_eye, device=self.device)
            self.val_camera.append(temp_val_camera)
        
        self.video_camera, _, _ = get_camera(
            1, self.camera_target, self.r,
            self.r * torch.tensor([[0, 0, -1]]),
            device=self.device
        )

    def _rotate_mesh(self, mesh, theta):
        """Rotate mesh around Y-axis by theta radians."""
        rotation_matrix = torch.tensor([
            [theta.cos(), 0, -theta.sin()],
            [          0, 1,            0],
            [theta.sin(), 0,  theta.cos()]
        ]).to(self.device)
        mesh.vertices = torch.matmul(rotation_matrix, mesh.vertices.unsqueeze(-1)).squeeze()

    def _compute_adaptive_timestep(self, min_step, max_step, iter_frac):
        """
        Compute adaptive timestep range following HiFA strategy.
        """
        step = max_step - (max_step - min_step) * math.sqrt(iter_frac)
        return int(step)

    def _clamp_textures(self):
        """Clamp texture values to physically valid ranges."""
        for key, clamp_range in [
            ('normals_texture', ((-0.5, 0.5,), (-0.5, 0.5), (0.5, 1.0))),
            ('diffuse_texture', ((0., 1.), (0., 1.), (0., 1.))),
            ('metallic_texture', ((0., 1.),)),
            ('roughness_texture', ((0., 1.),)),
        ]:
            if key in self.texture:
                if key == 'normals_texture':
                    self.texture[key].data = F.normalize(self.texture[key].data, dim=-1)
                for d, clamp_range_d in enumerate(clamp_range):
                    self.texture[key].data[..., d].clamp_(*clamp_range_d)

    def training_step(self, timestep):

        iter_frac = timestep / self.num_training_steps
        t_min = self._compute_adaptive_timestep(200, 300, iter_frac)
        t_max = self._compute_adaptive_timestep(500, 980, iter_frac)

        rand_scale = torch.rand(1) * (self.max_radius_scale - self.min_radius_scale) + self.min_radius_scale
        camera, camera_thetas, camera_phis = get_camera(
            self.batch_size, self.camera_target, rand_scale * self.r, device=self.device
        )

        view_dirs = get_view_direction(camera_thetas, camera_phis)
        if self.use_directional_embeddings:
            view_dir_embeddings = self.prompt_embeddings[0][view_dirs]
            repeated_embeddings = self.prompt_embeddings[1][0].repeat(self.batch_size, 1, 1)
            batch_embeddings = [view_dir_embeddings, repeated_embeddings]
        else:
            batch_embeddings = [
                self.prompt_embeddings[0].repeat(self.batch_size, 1, 1),
                self.prompt_embeddings[1].repeat(self.batch_size, 1, 1)
            ]

        rotate_theta = 2 * torch.pi * torch.rand(1)
        condition_image = None
        if self.stage == 'ii':
            with torch.no_grad():
                self._rotate_mesh(self.original_mesh, rotate_theta)
                condition_image = render(self.original_mesh, camera, self.light, random_background=True)
                condition_image = torch.movedim(condition_image, -1, 1)
                self._rotate_mesh(self.original_mesh, -rotate_theta)

        self._rotate_mesh(self.mesh, rotate_theta)
        image = render(self.mesh, camera, self.light, random_background=True)
        image = torch.movedim(image, -1, 1)
        self._rotate_mesh(self.mesh, -rotate_theta)

        loss = self.loss(
            image,
            condition_image,
            batch_embeddings,
            min_timestamp=t_min,
            max_timestamp=t_max,
            guidance_scale=self.guidance_scale,
            lowres_noise_level=self.lowres_noise_level if self.stage == 'ii' else None,
        )

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self._clamp_textures()
    

    def validation_step(self, current_step):
        """Render validation views and save visualizations."""

        # Plot rendered views
        imgs = render(self.mesh, self.val_camera, self.val_light, val_background=True)   
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        val_dirs = get_view_direction(self.val_thetas.squeeze(), self.val_phis.squeeze())        
        for i, ax in enumerate(axes.flat):
            ax.imshow(imgs[i].clamp(0., 1.).detach().cpu())
            ax.set_title(self.dir2name[val_dirs[i].item()])
            ax.axis('off')    
        render_path = os.path.join(
            self.experiment_path,
            f'render_stage_{self.stage}_step_{current_step}.png'
        )
        plt.savefig(render_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot texture maps
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        texture_keys = ['diffuse_texture', 'normals_texture', 'metallic_texture', 'roughness_texture']
        for ax, key in zip(axes.flat, texture_keys):
            if key in self.texture.keys():
                data = self.texture[key].detach().cpu()
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                ax.imshow(data)
            ax.axis('off')
            ax.set_title(key.replace('_texture', ''))
        texture_path = os.path.join(
            self.experiment_path,
            f'texture_stage_{self.stage}_step_{current_step}.png'
        )
        plt.savefig(texture_path, bbox_inches='tight', dpi=300)
        plt.close()

    def make_video(self, current_step):
        """Generate 360 rotation video."""
        video_path = os.path.join(
            self.experiment_path,
            f"rotation_stage_{self.stage}_step_{current_step}.mp4"
        )
        
        writer = imageio.get_writer(video_path, fps=30)
        num_frames = 120
        thetas = torch.arange(0., 2 * torch.pi, 2 * torch.pi / num_frames)
        
        for theta in thetas:
            self._rotate_mesh(self.mesh, theta)
            image = render(self.mesh, self.video_camera, self.val_light, val_background=True)
            self._rotate_mesh(self.mesh, -theta)
            
            image_cpu = image[0].clamp(0., 1.).detach().cpu()
            image_int = (255 * image_cpu).numpy().astype(np.uint8)
            writer.append_data(image_int)
        
        writer.close()
        print(f"Video saved to {video_path}")

    def run_training_loop(self):
        print(f"\nTraining for {self.num_training_steps} steps...")
        
        for i in tqdm(range(self.num_training_steps), desc=f"Stage {self.stage}"):
            self.training_step(i)
            
            if i % self.eval_plot_iter == 0:
                self.validation_step(i)
        
        self.validation_step(self.num_training_steps)
        self.make_video(self.num_training_steps)
        
        # Save textured mesh
        model_dir = os.path.join(self.experiment_path, f'model_stage_{self.stage}')
        os.makedirs(model_dir, exist_ok=True)
        write_obj(model_dir, self.mesh, self.texture)     

        print(f"Stage {self.stage} training completed!")