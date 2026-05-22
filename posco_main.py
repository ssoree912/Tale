import os
import cv2
import torch
import argparse
import json
import shutil
import re
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from accelerate import Accelerator 
from diffusers.schedulers import DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler


from pipeline_tale_pixart_alpha import TALEPixArtAlphaPipeline
from modeling_t5_exceptional import T5EncoderModelExceptional
from pipeline_tale_stable_diffusion import TALEStableDiffusionPipeline
from modeling_clip_exceptional import CLIPTextModelExceptional     
from clip.base_clip import CLIPEncoder
from utils import load_bg, load_fg

def get_crop_box(mask_path, padding_factor=0.2):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    

    rows = np.any(mask_np > 128, axis=1)
    cols = np.any(mask_np > 128, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return 0, 0, mask.width, mask.height
        
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    w = x_max - x_min
    h = y_max - y_min
    

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    

    size = int(max(w, h) * (1 + padding_factor))
    

    half_size = size // 2
    
    x1 = cx - half_size
    y1 = cy - half_size
    x2 = cx + half_size
    y2 = cy + half_size
    
    return x1, y1, x2, y2

def crop_and_pad(img, box):
    x1, y1, x2, y2 = box
    w, h = img.size
    
    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(w, x2)
    iy2 = min(h, y2)
    

    crop = img.crop((ix1, iy1, ix2, iy2))

    target_w = x2 - x1
    target_h = y2 - y1

    bg_color = 0 if img.mode == "L" else (0, 0, 0)

    new_img = Image.new(img.mode, (target_w, target_h), bg_color)
    
    paste_x = ix1 - x1
    paste_y = iy1 - y1
    new_img.paste(crop, (paste_x, paste_y))
    
    return new_img

def load_metadata_prompts(data_dir):
    prompts = {}
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        return prompts
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample = record.get("sample")
            prompt = record.get("prompt")
            if sample and prompt:
                prompts[sample] = prompt
    return prompts


def prompt_for_sample(sample, metadata_prompts, default_prompt):
    if sample in metadata_prompts:
        return metadata_prompts[sample]
    parts = sample.split(" ", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    return default_prompt


SAMPLE_NUMBER_PATTERN = re.compile(r"(?:^|_)(\d{6,7})(?:\D*$|$)")


def sample_sort_key(item):
    sample_key, sample_name, _, _ = item
    matches = SAMPLE_NUMBER_PATTERN.findall(sample_name)
    sample_number = int(matches[-1]) if matches else float("inf")
    return sample_number, sample_key


def discover_samples(data_dir):
    samples = []
    required = {"background.png", "foreground.png", "segmentation.png", "location.png"}
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        if required.issubset(set(files)):
            sample_dir = root
            rel_dir = os.path.relpath(sample_dir, data_dir)
            sample_name = os.path.basename(sample_dir)
            rel_parent = os.path.dirname(rel_dir)
            if rel_parent == ".":
                rel_parent = ""
            sample_key = os.path.join(rel_parent, sample_name) if rel_parent else sample_name
            samples.append((sample_key, sample_name, rel_parent, sample_dir))
            dirs[:] = []
    return sorted(samples, key=sample_sort_key)


def collect_existing_outputs(output_dir, flat_output, output_ext):
    existing = {}
    if not os.path.isdir(output_dir):
        return existing

    ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
    ext = ext.lower()
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        if flat_output:
            for name in files:
                if name.startswith("."):
                    continue
                stem, suffix = os.path.splitext(name)
                if suffix.lower() == ext:
                    existing.setdefault(stem, os.path.join(root, name))
        elif "results_highres.png" in files:
            existing.setdefault(os.path.basename(root), os.path.join(root, "results_highres.png"))
    return existing


def main():
    parser = argparse.ArgumentParser(description='Parameters for running TALE framework')
    parser.add_argument('--model_path', type=str, default='./stable-diffusion-2-1-base')
    parser.add_argument('--data_dir', type=str, default='examples')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--tprime', type=int, default=12)
    parser.add_argument('--tau', type=int, default=5)
    parser.add_argument('--inv_guidance_scale', type=float, default=5.)
    parser.add_argument('--comp_guidance_scale', type=float, default=10.)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--crop_padding', type=float, default=0.5, help='Padding ratio for cropping context')
    parser.add_argument('--flat_output', action='store_true', help='Save output_dir/<relative>/<sample>.jpg instead of output_dir/<relative>/<sample>/results_highres.png')
    parser.add_argument('--output_ext', type=str, default='.jpg')
    parser.add_argument('--default_prompt', type=str, default='an industrial object on a railway track at a steel mill')
    parser.add_argument('--skip-existing', '--skip_existing', dest='skip_existing', action='store_true', help='Skip samples whose output image already exists')

    args = parser.parse_args()

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
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    clip = CLIPEncoder(device)
    pipe.clip = clip

    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
    inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")

    image_h, image_w = (512, 512) 

    metadata_prompts = load_metadata_prompts(args.data_dir)
    samples = discover_samples(args.data_dir)
    ext = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"
    existing_outputs = collect_existing_outputs(args.output_dir, args.flat_output, ext) if args.skip_existing else {}
    if args.skip_existing:
        print(f"existing_outputs={len(existing_outputs)}")
    os.makedirs(args.output_dir, exist_ok=True)
    for k, (sample_key, sample, rel_parent, sample_dir) in enumerate(samples):
        prompt = prompt_for_sample(sample, metadata_prompts, args.default_prompt)
        output_base_dir = os.path.join(args.output_dir, rel_parent) if rel_parent else args.output_dir
        if args.flat_output:
            result_dir = os.path.join(output_base_dir, f".tmp_{sample}")
            output_path = os.path.join(output_base_dir, f"{sample}{ext}")
        else:
            result_dir = os.path.join(output_base_dir, sample)
            output_path = os.path.join(result_dir, "results_highres.png")

        existing_output = existing_outputs.get(sample)
        if args.skip_existing and (os.path.exists(output_path) or existing_output):
            print(f"[skip-existing] {sample_key}: {existing_output or output_path}")
            continue

        os.makedirs(result_dir, exist_ok=True)

        bg_path = os.path.join(sample_dir, "background.png")
        fg_path = os.path.join(sample_dir, "foreground.png")
        seg_path = os.path.join(sample_dir, "segmentation.png")
        mask_path = os.path.join(sample_dir, "location.png")


        orig_bg_pil = Image.open(bg_path).convert("RGB")
        orig_mask_pil = Image.open(mask_path).convert("L")
        

        crop_box = get_crop_box(mask_path, padding_factor=args.crop_padding)
        print("crop_box:", crop_box, "orig_size:", orig_bg_pil.size)
        

        cropped_bg = crop_and_pad(orig_bg_pil, crop_box)
        cropped_mask = crop_and_pad(orig_mask_pil, crop_box)
        
        resized_bg = cropped_bg.resize((image_w, image_h), Image.LANCZOS)
        resized_mask = cropped_mask.resize((image_w, image_h), Image.NEAREST)

        temp_bg_path = os.path.join(result_dir, "temp_bg.png")
        temp_mask_path = os.path.join(result_dir, "temp_mask.png")
        
        resized_bg.save(temp_bg_path)
        resized_mask.save(temp_mask_path)
        
        
        pipe.clip.encode_ref(temp_bg_path, prompt)

        bg_img = load_bg(temp_bg_path, (image_w, image_h))
        bg_latents = pipe.img_to_latents(bg_img).to(device)
        latent_h, latent_w = bg_latents.shape[2:]

        fg_img, seg, img_bbox, latent_bbox, ref_bbox = load_fg(
            fg_path, temp_mask_path, seg_path, (image_w, image_h), (latent_w, latent_h)
        )
        
        all_fg_img = [fg_img]
        all_fg_latents = [pipe.img_to_latents(fg_img).to(device) for fg_img in all_fg_img]

        # --- Inversion ---
        with torch.no_grad():
            pipe.scheduler = inverse_scheduler
            inverted_bg_latents, inv_prompt_inputs = pipe(
                "", negative_prompt="", num_inference_steps=args.num_inference_steps,
                guidance_scale=args.inv_guidance_scale, latents=bg_latents, output_type="latent"
            )
            (inv_prompt_embeds, inv_prompt_attention_mask, inv_neg_embeds, inv_neg_mask) = inv_prompt_inputs

            all_inverted_fg_latents = [
                pipe(
                    prompt_embeds=inv_prompt_embeds, prompt_attention_mask=inv_prompt_attention_mask,
                    negative_prompt_embeds=inv_neg_embeds, negative_prompt_attention_mask=inv_neg_mask,
                    num_inference_steps=args.num_inference_steps, guidance_scale=args.inv_guidance_scale,
                    latents=in_latents, output_type="latent"
                )[0] for in_latents in all_fg_latents
            ]        

        pipe.scheduler = scheduler
        inverted_latents = torch.cat([inverted_bg_latents] + all_inverted_fg_latents)

        # --- Composition ---
        out = pipe(
            prompt=prompt, negative_prompt="", num_inference_steps=args.num_inference_steps,
            guidance_scale=args.comp_guidance_scale, latents=inverted_latents, composition=True,
            inv_prompt_embeds=inv_prompt_embeds, inv_prompt_attention_mask=inv_prompt_attention_mask,
            inv_negative_prompt_embeds=inv_neg_embeds, inv_negative_prompt_attention_mask=inv_neg_mask,
            inv_guidance_scale=args.inv_guidance_scale,
            obj_params=[dict(ref_bbox=ref_bbox, latent_bbox=latent_bbox, img_bbox=img_bbox, seg=seg)],
            tprime=args.tprime, tau=args.tau,
        )[0]    

        generated_crop = out[0]

        crop_w = crop_box[2] - crop_box[0]
        crop_h = crop_box[3] - crop_box[1]
        
        restored_crop = generated_crop.resize((crop_w, crop_h), Image.LANCZOS)
        
        paste_x = max(0, crop_box[0])
        paste_y = max(0, crop_box[1])
        
        crop_start_x = paste_x - crop_box[0]
        crop_start_y = paste_y - crop_box[1]
        
        paste_w = min(orig_bg_pil.width, crop_box[2]) - paste_x
        paste_h = min(orig_bg_pil.height, crop_box[3]) - paste_y
        
        if paste_w > 0 and paste_h > 0:
            final_patch = restored_crop.crop(
                (crop_start_x, crop_start_y, crop_start_x + paste_w, crop_start_y + paste_h)
            )

            mask_patch = orig_mask_pil.crop((paste_x, paste_y, paste_x + paste_w, paste_y + paste_h))

            mask_patch = mask_patch.point(lambda p: 255 if p > 128 else 0)

            orig_bg_pil.paste(final_patch, (paste_x, paste_y), mask_patch)

        orig_bg_pil.save(output_path)


        
        if os.path.exists(temp_bg_path): os.remove(temp_bg_path)
        if os.path.exists(temp_mask_path): os.remove(temp_mask_path)
        if args.flat_output:
            shutil.rmtree(result_dir, ignore_errors=True)
        
        print(f"Processed {sample_key}: Saved high-res result to {output_path}.")

if __name__ == "__main__":
    main()