import torch
import math
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import imageio
import torch.nn.functional as F
from torchvision.transforms import v2
import random
import envlight
from collections import defaultdict

from our_kaolin import rot33_rotate
from camera import get_camera

from renderer import render
from guidance import SDSLoss, SDSDSLoss, SDILoss
# from guidance_full import SDSLoss, SDILoss
from regularizers import reg1, reg2, reg3, smoothness_regularizer
from prompt_processing import get_view_direction
from mesh import copy_mesh, write_obj
from torch.nn import Parameter, ParameterDict
from torch.optim.lr_scheduler import LambdaLR

class Trainer:

    def __init__(self, mesh, texture, prompt_embeddings, stage, experiment_path, config):
        
        self.batch_size = config.batch_size
        self.experiment_path = experiment_path
        self.device = 'cuda:0'
        self.mesh = mesh
        self.camera_target, self.r = self.get_camera_parameters()
        self.texture = texture
        self.prompt_embeddings = prompt_embeddings
        self.objaverse_eval = config.objaverse_eval
        self.eval_plot_iter = config.eval_plot_iter
        self.eval_renders_count = config.eval_renders_count
        self.lowres_noise_level = config.lowres_noise_level
        self.use_random_background = not config.no_random_background
        self.use_directional_embeddings = config.use_directional_embeddings
        self.dir2name = {
            0 : 'front view',
            1 : 'side view',
            2 : 'backside view',
            3 : 'top view', 
            4 : 'bottom view',
        }
        self.gs_limit = config.gs_limit
        self.gamma_correction_1 = config.gamma_correction_1
        self.gamma_value_1 = config.gamma_value_1
        self.gamma_correction_2 = config.gamma_correction_2
        self.gamma_value_2 = config.gamma_value_2
        self.clip_x0 = not config.no_x0_clipping
        self.mesh_rotation = not config.no_mesh_rotation

        self.stage = stage
        if self.stage == 'i_sd':
            self.n_training_steps = config.n_steps_1
            self.loss = SDSDSLoss()
            self.guidance_scale = config.guidance_scale_1
            self.lr = config.lr_1
            self.light = envlight.EnvLight(
                config.env_light_1,
                scale=config.scale_light_1,
                device='cuda'
            )
            self.schedule = self.set_linear_time_strategy(
                self.n_training_steps, config.min_noise_level_1, config.max_noise_level_1)
            self.use_gs_schedule = config.use_gs_schedule_1
            self.use_sds_schedule = config.use_time_schedule_1
        else:
            self.use_sds_schedule = config.use_time_schedule_1
            self.n_training_steps = config.n_steps_1
            self.lr = config.lr_1
            
            model_name = config.model_i if self.stage == 'i' else config.model_ii
            print(f"\nFor stage {self.stage} we use model {model_name}.\n")
            self.loss = SDILoss(self.stage, model_name=model_name)
            
            self.guidance_scale = config.guidance_scale_1
            self.light = envlight.EnvLight(
                config.env_light_1,
                scale=config.scale_light_1,
                device='cuda'
            )
            self.schedule = self.set_linear_time_strategy(
                self.n_training_steps, config.min_noise_level_1, config.max_noise_level_1)
            self.use_gs_schedule = config.use_gs_schedule_1

        if self.stage == 'ii':
            self.use_sds_schedule = config.use_time_schedule_2
            self.n_training_steps = config.n_steps_2
            self.lr = config.lr_2
            self.original_mesh = copy_mesh(mesh.detach(), texture)
            self.original_mesh.materials[0].FG_LUT = self.mesh.materials[0].FG_LUT
            self.guidance_scale = config.guidance_scale_2
            self.light = envlight.EnvLight(
                config.env_light_2,
                scale=config.scale_light_2,
                device='cuda'
            )
            self.schedule = self.set_linear_time_strategy(
                self.n_training_steps, config.min_noise_level_2, config.max_noise_level_2)
            self.use_gs_schedule = config.use_gs_schedule_2
        
        self.light_name = config.env_light_2
        self.random_light_scale = config.random_light_scale
        self.val_light = envlight.EnvLight(
            config.env_light_2,
            scale=2.,
            device='cuda'
        )

        self.gs_schedule = self.set_linear_time_strategy(
            self.n_training_steps, 5, self.guidance_scale)

        if (config.optimizer_name == 'Adam'):
            self.optimizer = torch.optim.Adam(
                texture.parameters(), 
                lr=self.lr
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=config.gamma_scheduler)
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=1) #никак не обновляем lr

            # Cosine sheduler
            # min_lr = 1e-5  # Minimum learning rate
            # import math
            # def lr_lambda(iter):
            #     progress = iter / self.n_training_steps
            #     cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            #     return max(cosine_decay, min_lr / self.lr)

            # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)


            # Decay scheduler
            # decay_rate = 0.999  # Adjust for faster/slower decay
            # def lr_lambda(iter):
            #     return decay_rate ** iter

            # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            raise NotImplementedError()


        print(f'Running training stage {stage}')
        if not self.objaverse_eval:
            self.val_phis = torch.linspace(0, 2 * torch.pi, self.eval_renders_count+1)[:-1].view(1, -1)
            # mishan: make more general
            self.val_thetas = torch.zeros(1, self.eval_renders_count)
        else:
            self.val_phis = torch.Tensor((
                [0.5 * math.pi * (0.5 + i) for i in range(4)]
                + [0.25 * math.pi * i for i in range(8)]
                + [0.25 * math.pi * i for i in range(8)]
            )).view(1, -1)
            self.val_thetas = torch.Tensor(4 * [0,] + 8 * [math.pi / 6] + 8 * [math.pi / 3]).view(1, -1)

        self.val_camera = []
        for i in range(0, len(self.val_phis[0]), 10):
            phi = self.val_phis[0][i:i+10]
            theta = self.val_thetas[0][i:i+10]

            val_eye = self.r * torch.stack([
                phi.cos() * theta.cos(),
                torch.ones_like(phi) * theta.sin(),
                phi.sin() * theta.cos()
            ], dim=-1).view(-1, 3)
            temp_val_camera, _, _ = get_camera(len(val_eye), self.camera_target, self.r, val_eye)
            self.val_camera.append(temp_val_camera)

        self.video_camera, _, _ = get_camera(1, self.camera_target, self.r, self.r * torch.tensor([[0,0,-1]]))
    
    def set_time_strategy(self, output_shape, diffusion_steps=1000):
        def new_weights(t, m1=800, m2=500, s1=300, s2=100):
            return (
                torch.exp(-(t - m1) * (t - m1) / 2 / s1 / s1) * (t > m1) +
                1 * ((t >= m2) & (t <= m1)) +
                torch.exp(-(t - m2) * (t - m2) / 2 / s2 / s2) * (t < m2)
            )
        w = new_weights(torch.arange(1, diffusion_steps+1))
        weights_sum = w.sum()
        weights_cumsum = torch.cumsum(w.flip(dims=(0,)), dim=0).flip(dims=(0,))
        timesteps = (diffusion_steps - 1 - torch.searchsorted(
            (weights_cumsum / weights_sum).flip(dims=(0,)),
            torch.arange(1, output_shape+1) / output_shape
        )).to(torch.long)
        return timesteps
    
    def set_linear_time_strategy(self, output_shape, min_diffusion_steps=20, max_diffusion_steps=980):
        return torch.linspace(min_diffusion_steps, max_diffusion_steps, output_shape).flip(0).to(torch.long)
    
    def get_camera_parameters(self):
        camera_target = torch.tensor([0., 0., 0.])
        r = 2 * torch.tensor(2.2)
        return camera_target, r

    def rotate_mesh(self, mesh, theta):
        rotation_matrix = torch.tensor([
            [theta.cos(), 0, -theta.sin()],
            [          0, 1,            0],
            [theta.sin(), 0,  theta.cos()]
        ]).cuda()
        mesh.vertices = torch.matmul(rotation_matrix, mesh.vertices.unsqueeze(-1)).squeeze()

    def compute_step(self, min_step, max_step, iter_frac):
        # From work HiFA
        step = (max_step - (max_step - min_step) * math.sqrt(iter_frac))
        return int(step)

    # def update_light(self):
    #     temp_scale = torch.rand(1) * 1 + 1
    #     self.light = envlight.EnvLight(
    #             self.light_name,
    #             scale=temp_scale.item(),
    #             device='cuda'
    #         )
    
    def training_step(self, timestep):
        # mishan: take from paint-it, numbers from paint-it
        t_min = self.compute_step(200, 300, timestep/self.n_training_steps)
        t_max = self.compute_step(500, 980, timestep/self.n_training_steps)

        # if self.random_light_scale:
        #     self.update_light()

        # sample camera
        min_radius_scale = 0.5
        max_radius_scale = 1.5
        rand_scale = torch.rand(1) * (max_radius_scale - min_radius_scale) + min_radius_scale
        camera, camera_thetas, camera_phis = get_camera(self.batch_size, self.camera_target, rand_scale * self.r)
        # mishan: add positional encodings
        view_dirs = get_view_direction(camera_thetas, camera_phis)
        if self.use_directional_embeddings:
            # batch_embeddings = [prompt_embeddings[0][i] for i in view_dirs] + \
            #                     [prompt_embeddings[1].repeat(self.batch_size, 1, 1)]
            view_dir_embeddings = self.prompt_embeddings[0][view_dirs]
            repeated_embeddings = self.prompt_embeddings[1][0].repeat(self.batch_size, 1, 1)
            batch_embeddings = [view_dir_embeddings, repeated_embeddings]
        else:
            # batch_embeddings = (conditioned, unconditioned for cfg)
            batch_embeddings = [self.prompt_embeddings[0].repeat(self.batch_size, 1, 1),
                                self.prompt_embeddings[1].repeat(self.batch_size, 1, 1)]

        # random model rotation used to diversify lighting conditions
        theta = 2 * torch.pi * torch.rand(1)
        condition_image = None
        if self.stage == 'ii':
            with torch.no_grad():
                if self.mesh_rotation:
                    self.rotate_mesh(self.original_mesh, theta)
                condition_image = render(self.original_mesh, camera, self.light)
                condition_image = torch.movedim(condition_image, -1, 1)
                if self.mesh_rotation:
                    # do I really need to rotate verices back?
                    self.rotate_mesh(self.original_mesh, -theta)

        # TODO: render normal maps only?
        if self.mesh_rotation:
            self.rotate_mesh(self.mesh, theta)


        image = render(self.mesh, camera, self.light, random_background=self.use_random_background)
        image = torch.movedim(image, -1, 1)
        # if self.gamma_correction and self.stage == 'i':
        if self.gamma_correction_1 and self.stage == 'i':
            image = torch.pow(image, 1 / self.gamma_value_1)
        elif self.gamma_correction_2 and self.stage == 'ii':
            image = torch.pow(image, 1 / self.gamma_value_2)

        if self.mesh_rotation:
            self.rotate_mesh(self.mesh, -theta)

        if self.use_sds_schedule:
            scheduler_timestep = self.schedule[timestep]
        else:
            scheduler_timestep = None

        if self.use_gs_schedule:
            temp_guidance_scale = self.gs_schedule[timestep]
        else:
            temp_guidance_scale = self.guidance_scale

        loss = self.loss(
            image, 
            batch_embeddings, 
            condition_image, 
            min_step=t_min, 
            max_step=t_max, 
            guidance_scale=temp_guidance_scale,
            lowres_noise_level=self.lowres_noise_level,
            scheduler_timestep = scheduler_timestep,
            gs_limit = self.gs_limit,
            clip_x0=self.clip_x0
        )

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # clamp textures:
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

    def validation_step(self, current_step):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        image = torch.Tensor([]).to(self.device)
        for i in range(len(self.val_camera)):
            image = torch.cat((image, render(self.mesh, self.val_camera[i], self.val_light, val_background=True)))
        # image = render(self.mesh, self.val_camera, self.light, random_background=False)
        
        val_dirs = get_view_direction(self.val_thetas[0], self.val_phis[0])
        for i, ax in enumerate(axes.flat):
            ax.imshow(image[i].clamp(0., 1.).detach().cpu())
            ax.set_title(self.dir2name[val_dirs[i].item()])
            ax.axis('off')
        log_filename = os.path.join(self.experiment_path, f'render_stage_{self.stage}_step_{current_step:04d}.png')
        plt.savefig(log_filename, bbox_inches='tight', dpi=300)
        plt.close()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
        log_filename = os.path.join(self.experiment_path, f'texture_stage_{self.stage}_step_{current_step:04d}.png')
        for ax, key in zip(axes.flat, ['diffuse_texture', 'normals_texture', 'metallic_texture', 'roughness_texture']):
            if key in self.texture.keys():
                data_to_plot = self.texture[key].detach().cpu()
                data_to_plot -= data_to_plot.min()
                data_to_plot /= (data_to_plot.max() + 1e-8)
                ax.imshow(data_to_plot)
            ax.axis('off')
            ax.set_title(key)
        plt.savefig(log_filename, bbox_inches='tight', dpi=300)
        plt.close()


    def make_video(self, current_step):
        video_filename = os.path.join(
            self.experiment_path,
            f"light_motion_stage_{self.stage}_step_{current_step}.mp4"
        )
        writer = imageio.get_writer(video_filename, fps=30)
        num_frames = 120 # mishan: increase this value 2 times
        thetas = torch.arange(0., 2 * torch.pi, 2 * torch.pi / num_frames)
        for i, theta in enumerate(thetas):
            self.rotate_mesh(self.mesh, theta)
            image = render(self.mesh, self.video_camera, self.val_light, val_background=True)
            self.rotate_mesh(self.mesh, -theta)
            image_cpu = image[0].clamp(0., 1.).detach().cpu()
            image_int = (255 * image_cpu).numpy().astype(np.uint8)
            writer.append_data(image_int)
        writer.close()

    def final_step(self, current_step):
        if self.stage == 'i_sd' or self.objaverse_eval:

            render_dir = os.path.join(self.experiment_path, f'renders_stage_{self.stage}')
            os.mkdir(render_dir)

            # image = render(self.mesh, self.val_camera, self.light, random_background=False)
            image = torch.Tensor([]).to(self.device)
            for i in range(len(self.val_camera)):
                image = torch.cat((image, render(self.mesh, self.val_camera[i], self.val_light, random_background=False)))
            
            for i in range(len(image)):
                plt.imshow(image[i].clamp(0., 1.).detach().cpu())
                plt.axis('off')
                log_filename = os.path.join(render_dir, f'render_stage_{self.stage}_{i}.png')
                plt.savefig(log_filename, bbox_inches='tight', dpi=300)
                plt.close()
        self.make_video(current_step)
        # save model
        model_dir = os.path.join(self.experiment_path, f'model_stage_{self.stage}')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        write_obj(model_dir, self.mesh, self.texture)

    def run_training_loop(self):
        # mishan: change it to make objaverse work
        for i in tqdm(range(self.n_training_steps)):
            self.training_step(i)
            if not self.objaverse_eval and i % self.eval_plot_iter == 0:
                self.validation_step(i)
            # if self.stage == 'ii' and i == 500:
            #     break
        if not self.objaverse_eval:
            self.validation_step(self.n_training_steps)
        self.final_step(self.n_training_steps)
