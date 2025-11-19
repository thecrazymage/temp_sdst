import os
import torch

from transformers import T5EncoderModel
from diffusers import DiffusionPipeline,  StableDiffusionPipeline

def get_prompt_filename(prompt, directional=True):
    if directional:
        return f'directional_embeddings_{prompt.replace(" ", "_")}.pt'
    else:
        return f'embeddings_{prompt.replace(" ", "_")}.pt'

def get_view_direction(thetas, phis):
    '''
    Classifies camera positions into different views:
    0 - front view, 1 - side view, 2 - backside view, 3 - top view, 4 - bottom view
    '''
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    phis = phis % (2 * torch.pi)
    thetas = thetas % (2 * torch.pi)

    # Front view: phi in [pi/4, 3pi/4]
    front_mask = (phis >= torch.pi / 4) & (phis <= 3 * torch.pi / 4)

    # Backside view: phi in [5pi/4, 7pi/4]
    backside_mask = (phis >= 5 * torch.pi / 4) & (phis <= 7 * torch.pi / 4)

    # Side view: phi in (-pi/4, pi/4) or (3pi/4, 5pi/4)
    side_mask = ((phis > 7 * torch.pi / 4) | (phis < torch.pi / 4)) | ((phis > 3 * torch.pi / 4) & (phis < 5 * torch.pi / 4))

    # Top view: theta in [pi/4, 3pi/4]
    top_mask = (thetas >= torch.pi / 4) & (thetas <= 3 * torch.pi / 4)

    # Bottom view: theta in [5pi/4, 7pi/4]
    bottom_mask = (thetas >= 5 * torch.pi / 4) & (thetas <= 7 * torch.pi / 4)

    res[front_mask] = 0
    res[side_mask] = 1
    res[backside_mask] = 2
    res[top_mask] = 3
    res[bottom_mask] = 4

    return res

def encode_prompt(prompt, directional=False, deepfloyd_model='DeepFloyd/IF-I-XL-v1.0'):
    if not os.path.isdir('cached_prompts'):
        os.mkdir('cached_prompts')
    embedding_filename = 'cached_prompts/' + get_prompt_filename(prompt, directional)
    if os.path.exists(embedding_filename):
        return torch.load(embedding_filename)
    
    #deepfloyd_model = "/home/maliev/.cache/huggingface/hub/models--DeepFloyd--IF-I-XL-v1.0/snapshots/c03d510e9b75bce9f9db5bb85148c1402ad7e694/"
    text_encoder = T5EncoderModel.from_pretrained(
        deepfloyd_model,
        subfolder="text_encoder",
        load_in_8bit=True,
        variant="8bit",
        # device_map={"": 0}, # так закидываем все на на 'cuda:0'
    )#.to('cuda:0')
    pipe = DiffusionPipeline.from_pretrained(
        deepfloyd_model,
        text_encoder=text_encoder, # pass the previously instantiated 8bit text encoder
        unet=None,
    )#.to('cuda:0')

    if directional:
        directions = ['front view', 'side view', 'backside view', 'top view', 'bottom view']
        prompts = [prompt + ', ' + direction for direction in directions]
    else:
        prompts = [prompt,]

    prompt_embeddings = pipe.encode_prompt(prompts)

    torch.save(prompt_embeddings, embedding_filename)
    return prompt_embeddings

def encode_prompt_sd(prompt, directional=False, model="runwayml/stable-diffusion-v1-5"):
    device = torch.device('cuda:0')
    if not os.path.isdir('cached_prompts'):
        os.mkdir('cached_prompts')
    embedding_filename = 'cached_prompts/' + get_prompt_filename('sd ' + prompt, directional)
    if os.path.exists(embedding_filename):
        return torch.load(embedding_filename)
    pipe = StableDiffusionPipeline.from_pretrained(
        model, 
        torch_dtype=torch.float16
        ).to(device)

    tokenizer = pipe.tokenizer
    encoder = pipe.text_encoder

    if directional:
        directions = ['front view', 'side view', 'top view', 'backside view']
        prompts = [prompt + ', ' + direction for direction in directions]
    else:
        prompts = [prompt,]

    inputs = tokenizer(prompts, padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt')
    prompt_embeddings_cond = encoder(inputs.input_ids.to(device))[0]

    inputs = tokenizer('', padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt')
    prompt_embeddings_uncond = encoder(inputs.input_ids.to(device))[0]

    prompt_embeddings = torch.cat((prompt_embeddings_cond, prompt_embeddings_uncond))

    torch.save(prompt_embeddings, embedding_filename)
    return prompt_embeddings
