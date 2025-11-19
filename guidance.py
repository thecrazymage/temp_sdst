import torch
import numpy as np
import math
import torch.nn.functional as F

from diffusers import DiffusionPipeline,  StableDiffusionPipeline, DDIMScheduler

class SDSLoss(torch.nn.Module):
    def __init__(self, stage='i', model_name="DeepFloyd/IF-I-XL-v1.0"):
        super().__init__()

        # load the model
        self.device = torch.device('cuda:0')
        self.stage = stage

        deepfloyd_model = model_name
        # if stage == 'i':
        #     deepfloyd_model = "DeepFloyd/IF-I-XL-v1.0"  # changed from XL
        #     # deepfloyd_model = "/home/maliev/.cache/huggingface/hub/models--DeepFloyd--IF-I-XL-v1.0/snapshots/c03d510e9b75bce9f9db5bb85148c1402ad7e694/"
        # elif stage == 'ii':
        #     deepfloyd_model = "DeepFloyd/IF-II-L-v1.0"  # changed from L
        #     # deepfloyd_model = "/home/maliev/.cache/huggingface/hub/models--DeepFloyd--IF-II-L-v1.0/snapshots/609476ce702b2d94aff7d1f944dcc54d4f972901/"
        # else:
        #     raise(NotImplementedError)
            
        pipe = DiffusionPipeline.from_pretrained(
            deepfloyd_model,
            text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)

        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.amp.autocast('cuda', enabled=False)
    def forward_unet(self, latents, t, encoder_hidden_states, **kwargs):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(torch.float16),
            t.to(torch.float16),
            encoder_hidden_states=encoder_hidden_states.to(torch.float16),
            **kwargs
        ).sample.to(input_dtype)

    def prepare_latents(self, images):
        resolution = (64, 64) if self.stage == 'i' else (256, 256)
        #resolution = (256, 256)
        latents = F.interpolate(images, resolution, mode="bilinear", align_corners=False, antialias=True)
        return 2. * latents - 1.

    def prepare_downscaled_latents(self, images, lowres_noise_level):
        downscaled = F.interpolate(images, (64, 64), mode="nearest")#, align_corners=False, antialias=True)
        upscaled = F.interpolate(downscaled, (256, 256), mode="nearest")#, align_corners=True).detach()
        upscaled = 2. * upscaled - 1.
        upscaled = self.scheduler.add_noise(
            upscaled,
            torch.randn_like(upscaled),
            torch.tensor(int(self.num_train_timesteps * lowres_noise_level))
        )
        return upscaled

    def construct_gradient(self, noise_pred, noise, t, guidance_scale):
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text, _ = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_text + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        #w = w * (1 - w) ** 0.5
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        return grad


    def forward(self, images, prompt_embeddings, original=None, min_step=20, max_step=980, guidance_scale=10., lowres_noise_level=0.75, scheduler_timestep=None):
        # prepare images
        batch_size = images.shape[0]
        latents = self.prepare_latents(images)
        if self.stage == 'ii':
            condition = images if original is None else original
            condition = self.prepare_downscaled_latents(condition, lowres_noise_level)
            noise_level = torch.full(
                    [2 * condition.shape[0]],
                    torch.tensor(int(self.num_train_timesteps * lowres_noise_level)),
                    device=condition.device
            )
        
        if scheduler_timestep is not None:
            t = scheduler_timestep * torch.ones(batch_size, dtype=torch.long, device=self.device)
        else:
            # sample ts
            t = torch.randint(
                min_step,
                max_step,
                [batch_size],
                dtype=torch.long,
                device=self.device
            )

        # predict noise
        with torch.no_grad():
            noise = torch.randn_like(latents, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            if self.stage == 'ii':
                latents_noisy = torch.cat([latents_noisy, condition], dim=1)
            latents_noisy = self.scheduler.scale_model_input(latents_noisy, t)
            if self.stage == 'i':
                noise_pred = self.forward_unet(
                    torch.cat(2 * [latents_noisy]),
                    torch.cat(2 * [t]),
                    torch.cat(prompt_embeddings),
                )
            else:
                noise_pred = self.forward_unet(
                    torch.cat(2 * [latents_noisy]),
                    torch.cat(2 * [t]),
                    torch.cat(prompt_embeddings),
                    class_labels=noise_level
                )
        # convert noise prediction into gradient
        grad = self.construct_gradient(noise_pred, noise, t, guidance_scale)
        # compute surrogate loss
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return loss_sds

class SDSDSLoss(torch.nn.Module):
    def __init__(self, input_type='rgb'):
        super().__init__()

        # load the model
        self.device = torch.device('cuda:0')

        model = "runwayml/stable-diffusion-v1-5"
            
        pipe = StableDiffusionPipeline.from_pretrained(
            model, 
            torch_dtype=torch.float16
        ).to(self.device)

        self.vae = pipe.vae.eval()
        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.alphas = self.scheduler.alphas_cumprod
        self.input_type = input_type
        self.scheduler.set_timesteps(self.num_train_timesteps, device=self.device)

    @torch.amp.autocast('cuda', enabled=False)
    def prepare_latents(self, images):
        resolution = (512, 512)
        images = F.interpolate(images, resolution, mode="bilinear", align_corners=False, antialias=True)
        images = 2. * images - 1.
        input_dtype = images.dtype
        posterior = self.vae.encode(images.to(torch.float16)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.amp.autocast('cuda', enabled=False)
    def decode_latents(self, latents):
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(torch.float16)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.amp.autocast('cuda', enabled=False)
    def forward_unet(self, latents, t, encoder_hidden_states, **kwargs):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(torch.float16),
            t.to(torch.float16),
            encoder_hidden_states=encoder_hidden_states.to(torch.float16),
            **kwargs,
        ).sample.to(input_dtype)

    def forward(self, images, prompt_embeddings, original=None, min_step=20, max_step=980, guidance_scale=10., lowres_noise_level=0.75, scheduler_timestep=None, **kwargs):
        # prepare images
        batch_size = images.shape[0]
        if self.input_type != 'latent':
            latents = self.prepare_latents(images)
        else:
            latents = images
        if scheduler_timestep is not None:
            t = scheduler_timestep * torch.ones(batch_size, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(
                min_step,
                max_step,
                [batch_size],
                dtype=torch.long,
                device=self.device
            )
        # predict noise
        with torch.no_grad():
            noise = torch.randn_like(latents, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latents_noisy = self.scheduler.scale_model_input(latents_noisy, t)
            noise_pred = self.forward_unet(
                torch.cat(2 * [latents_noisy]),
                torch.cat(2 * [t]),
                torch.cat(prompt_embeddings),
            )
        # convert noise prediction into gradient
        grad = self.construct_gradient(noise_pred, noise, t, guidance_scale)
        with torch.no_grad():
            decoded_image = self.decode_latents(latents)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return loss_sds

    def construct_gradient(self, noise_pred, noise, t, guidance_scale):
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1).to(noise_pred.dtype)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        #diff = noise_pred - noise
        return grad

# Currently, I use the class as an instance of SDS in x_0 parameterzation. The parameterization allows clamping target images and results in substantial improvement of texture fidelity in terms of output colors.
class SDILoss(SDSLoss):
    def forward(
        self,
        images,
        prompt_embeddings,
        original=None,
        min_step=20,
        max_step=980,
        guidance_scale=10.,
        invert_guidance_scale=0.,
        lowres_noise_level=0.75,
        scheduler_timestep=None,
        stochastic_inversion=True,
        gs_limit=None,
        clip_x0=True
    ):
        self.lowres_noise_level = lowres_noise_level
        self.stochastic_inversion = stochastic_inversion
        self.clip_x0 = clip_x0
        batch_size = images.shape[0]
        latents = self.prepare_latents(images)
        
        if scheduler_timestep is not None:
            t = scheduler_timestep * torch.ones(1, dtype=torch.long, device=self.device)
        else:
            # sample ts
            t = torch.randint(
                min_step,
                max_step,
                [1,],
                dtype=torch.long,
                device=self.device
            )
        
        # mishan: check cfg problem
        if gs_limit:
            guidance_scale = guidance_scale if t > gs_limit else 0 
            # print(guidance_scale)

        # predict noise
        with torch.no_grad():
            noise = torch.randn_like(latents, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            #latents_noisy, noise = self.invert(
            #    latents,
            #    prompt_embeddings,
            #    t,
            #    original=original,
            #    guidance_scale=invert_guidance_scale)
            noise_pred = self.predict_noise(
                latents_noisy,
                t,
                prompt_embeddings,
                original=original,
                guidance_scale=guidance_scale,
                lowres_noise_level=lowres_noise_level
            )
            latents_denoised = self.get_x0(latents_noisy, noise_pred, t)

        w = ((1 - self.alphas[t]) * self.alphas[t]).sqrt().view(-1, 1, 1, 1)
        self.debugging_stuff = {
            'latents'          : latents.detach(),
            'noise'            : noise.detach(),
            'latents_noisy'    : latents_noisy.detach(),
            'latents_denoised' : latents_denoised.detach(),
            't'                : t,
        }
        # TODO: This w leads to vanilla SDS. We can try other weighting strategies to improve the results. 
        loss_sds = 0.5 * w * F.mse_loss(latents, latents_denoised, reduction="sum") / batch_size
        return loss_sds

    def predict_noise(self, latents_noisy, current_t, prompt_embeddings, original, guidance_scale, lowres_noise_level):
        # Expand the latents if we are doing classifier free guidance
        batch_size = latents_noisy.shape[0]
        do_classifier_free_guidance = True
        if self.stage == 'ii':
            # TODO: hardcoded stuff
            condition = images if original is None else original
            condition = self.prepare_downscaled_latents(condition, lowres_noise_level)
            condition = self.scheduler.scale_model_input(condition, current_t)  # here i changed from next_t to current_t
            noise_level = torch.full(
                    [2 * condition.shape[0]],
                    torch.tensor(int(self.num_train_timesteps * lowres_noise_level)),
                    device=condition.device
            )
            latents_noisy = torch.cat([latents_noisy, condition], dim=1)
        latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, current_t)  # here i changed from next_t to current_t
        # Predict the noise residual
        if self.stage == 'i':
            noise_pred = self.forward_unet(
                latent_model_input,
                current_t * torch.ones(2 * batch_size, dtype=torch.long, device=self.device),
                torch.cat(prompt_embeddings),
            )
        elif self.stage == 'ii':
            noise_pred = self.forward_unet(
                latent_model_input,
                current_t * torch.ones(2 * batch_size, dtype=torch.long, device=self.device),
                torch.cat(prompt_embeddings),
                class_labels=noise_level,
            )
        else:
            raise NotImplementedError
        #classifier guidance:
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    def make_inversion_step(self, noise_pred, current_t, next_t, latents):
        alpha_t = self.alphas[current_t]
        alpha_t_next = self.alphas[next_t]
        if self.stochastic_inversion:
            sigma_t_next = ((1 - alpha_t) / (1 - alpha_t_next) * (1 - alpha_t_next / alpha_t)).sqrt()
        else:
            sigma_t_next = 0.
        sigma_t_next = 0.3 * (1 - alpha_t_next).abs().sqrt()
        pred_original_sample = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt())
        pred_original_sample = pred_original_sample.clamp(-1., 1.)
        latents = (
            pred_original_sample * alpha_t_next.sqrt()
            + (1 - alpha_t_next - sigma_t_next ** 2).abs().sqrt() * noise_pred
            + sigma_t_next * torch.randn_like(noise_pred)
        )
        return latents

    def get_inversion_timesteps(self, invert_to_t, batch_size):
        n_training_steps = 1000
        n_inversion_steps = 10
        t = invert_to_t[0].cpu().to(torch.int64).item()
        num_inference_steps = math.ceil(n_inversion_steps * t / 1000)
        num_inference_steps = max(5, num_inference_steps)
        timesteps = (
            np.linspace(0, t, num_inference_steps + 1)
            .round()#[::-1]
            .copy()
            .astype(np.int64)
        )
        return timesteps

    def get_noise_from_target(self, target, cur_xt, t):
        alpha_t = self.alphas[t]
        beta_t = 1 - alpha_t
        noise = (cur_xt - target * alpha_t ** (0.5)) / (beta_t ** (0.5))
        return noise

    def get_x0(self, original_samples, noise_pred, t):
        alpha_prod_t = self.alphas[t]
        beta_prod_t = 1 - alpha_prod_t
        x0 = (original_samples - noise_pred * beta_prod_t ** (0.5)) / (alpha_prod_t ** (0.5))
        if self.clip_x0:
            x0 = x0.clamp(-1., 1.)
        else:
            print("Hello __ 2")
        return x0
    
    def invert(
        self,
        start_latents,
        prompt_embeddings,
        t,
        guidance_scale,
        original,
        do_classifier_free_guidance=True
    ):
        latents = start_latents.clone()
        timesteps = self.get_inversion_timesteps(t, latents.shape[0])
        for current_t, next_t in zip(timesteps[:-1], timesteps[1:]):
            noise_pred = self.predict_noise(latents, current_t, prompt_embeddings, original, guidance_scale)
            latents = self.make_inversion_step(noise_pred, current_t, next_t, latents)
        found_noise = self.get_noise_from_target(start_latents, latents, next_t)
        return latents, found_noise

if __name__ == '__main__':
    from prompt_processing import encode_prompt
    prompt = "orange backpack"
    prompt_embeddings = encode_prompt(prompt)
    batch_embeddings = [prompt_embeddings[0].repeat(2, 1, 1),
                        prompt_embeddings[1].repeat(2, 1, 1)]

    images = torch.rand(2, 3, 64, 64, device=torch.device('cuda:0'))
    loss = SDILoss(stage='ii')
    loss(images, batch_embeddings, original=images.clone())
