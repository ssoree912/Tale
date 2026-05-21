import os
import cv2
import torch
import IPython
import argparse
import numpy as np

from pytorch_lightning import seed_everything
from PIL import Image, ImageDraw
from diffusers.schedulers import DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler

from pipeline_tale_pixart_alpha import TALEPixArtAlphaPipeline
from modeling_t5_exceptional import T5EncoderModelExceptional
from pipeline_tale_stable_diffusion import TALEStableDiffusionPipeline
from modeling_clip_exceptional import CLIPTextModelExceptional     

from clip.base_clip import CLIPEncoder
from utils import load_bg, load_fg

def main():
    parser = argparse.ArgumentParser(description='Parameters for running TALE framework')
    parser.add_argument('--model_path', type=str, default='./stable-diffusion-2-1-base', help='Path to the pretrained model') #PixArt-alpha/PixArt-XL-2-512x512, stabilityai/stable-diffusion-2-1-base
    parser.add_argument('--data_dir', type=str, default='examples', help='Directory to the input data')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save output images')
    parser.add_argument('--tprime', type=int, default=12, help='Value for selective tprime')
    parser.add_argument('--tau', type=int, default=5, help='Value for threshold tau')
    parser.add_argument('--inv_guidance_scale', type=float, default=5., help='Guidance scale for inversion')
    parser.add_argument('--comp_guidance_scale', type=float, default=10., help='Guidance scale for composition')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of steps for timesteps')

    args = parser.parse_args()

    ###############################################
    # Init pipeline
    ###############################################
    device = "cuda:0"
    if "PixArt" in args.model_path:
        pipe = TALEPixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        pipe.text_encoder = T5EncoderModelExceptional.from_pretrained(args.model_path, torch_dtype=torch.float16, subfolder="text_encoder")
        pipe.transformer.requires_grad_(False)
    elif "stable-diffusion" in args.model_path:
        pipe = TALEStableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        pipe.text_encoder = CLIPTextModelExceptional.from_pretrained(args.model_path, torch_dtype=torch.float16, subfolder="text_encoder")
        pipe.unet.requires_grad_(False)

    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    clip = CLIPEncoder(device)
    pipe.clip = clip


    ###############################################
    # Define schedulers for denoising and inversion
    ###############################################
    scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
    )

    inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
    )


    ###############################################
    # Conduct composition
    ###############################################
    for k, sample in enumerate(sorted(os.listdir(args.data_dir))):
        # Load inputs
        prompt = " ".join(sample.split(" ")[1:])
        sample_dir = os.path.join(args.data_dir, sample)

        bg = os.path.join(sample_dir, "background.png")
        fg = os.path.join(sample_dir, "foreground.png")
        seg = os.path.join(sample_dir, "segmentation.png")
        mask = os.path.join(sample_dir, "location.png")

        pipe.clip.encode_ref(bg, prompt)

        image_h, image_w = (512, 512)
        bg_img = load_bg(bg, (image_w, image_h))
        bg_latents = pipe.img_to_latents(bg_img).to(device)
        latent_h, latent_w = bg_latents.shape[2:]

        fg_img, seg, img_bbox, latent_bbox, ref_bbox = load_fg(fg, mask, seg, (image_w, image_h), (latent_w, latent_h))
        all_fg_img = [fg_img]
        all_fg_latents = [pipe.img_to_latents(fg_img).to(device) for fg_img in all_fg_img]

        # Stage 1: Image Inversion
        with torch.no_grad():
            pipe.scheduler = inverse_scheduler
            inverted_bg_latents, inv_prompt_inputs = pipe(
                "",
                negative_prompt="",
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.inv_guidance_scale,
                latents=bg_latents,
                output_type="latent"
            )

            (
                inv_prompt_embeds,
                inv_prompt_attention_mask,
                inv_negative_prompt_embeds,
                inv_negative_prompt_attention_mask
            ) = inv_prompt_inputs

            all_inverted_fg_latents = [
                pipe(
                    prompt_embeds=inv_prompt_embeds,
                    prompt_attention_mask=inv_prompt_attention_mask,
                    negative_prompt_embeds=inv_negative_prompt_embeds,
                    negative_prompt_attention_mask=inv_negative_prompt_attention_mask,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.inv_guidance_scale,
                    latents=in_latents,
                    output_type="latent"
                )[0] for in_latents in all_fg_latents
            ]        

        pipe.scheduler = scheduler
        inverted_latents = torch.cat([inverted_bg_latents] + all_inverted_fg_latents)

        # Stage 2: Image Composition
        out = pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.comp_guidance_scale,
            latents=inverted_latents,
            composition=True,
            inv_prompt_embeds=inv_prompt_embeds,
            inv_prompt_attention_mask=inv_prompt_attention_mask,
            inv_negative_prompt_embeds=inv_negative_prompt_embeds,
            inv_negative_prompt_attention_mask=inv_negative_prompt_attention_mask,
            inv_guidance_scale=args.inv_guidance_scale,
            obj_params=[
                dict(
                    ref_bbox=ref_bbox,
                    latent_bbox=latent_bbox,
                    img_bbox=img_bbox,
                    seg=seg,
                ),
            ],
            tprime=args.tprime,
            tau=args.tau,
        )[0]    

        # Save outputs
        target_w, target_h = 1920, 1080
        out_img = out[0].resize((target_w, target_h), Image.LANCZOS)
        result_dir = os.path.join(args.output_dir, sample)
        os.makedirs(result_dir, exist_ok=True)
        out_img.save(os.path.join(result_dir, "results.png"))

if __name__ == "__main__":
    main()