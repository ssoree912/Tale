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

def _butterworth_lowpass(shape, cutoff=0.1, order=4):
    """Butterworth low-pass mask in FFT-shifted coords.

    cutoff: normalized cutoff in (0, 1], measured as a fraction of the
        max radial frequency (i.e. the distance from DC to the nearest
        image edge in the fftshifted spectrum). cutoff=0.1 keeps the
        innermost 10% radius.
    order: filter order (higher = sharper transition).
    """
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(yy * yy + xx * xx)
    r_max = float(min(cy, cx))
    d0 = max(cutoff * r_max, 1e-6)
    return 1.0 / (1.0 + (r / d0) ** (2 * order))


def inject_lowfreq(generated_pil, reference_pil, alpha=1.0, cutoff=0.1, order=4, mask_pil=None, mask_feather=0):
    """Replace the low-frequency band of `generated` with that of `reference`
    using a Butterworth low-pass filter in the FFT domain.

    out_F = H * F(ref) + (1 - H) * F(gen)  where H is the Butterworth LP mask.
    With alpha < 1 the replacement is partial: H_eff = alpha * H.
    If `mask_pil` is given, only the area *outside* the mask receives the
    replacement (foreground object stays intact); the boundary is feathered
    in the spatial domain after IFFT to keep the FFT shift-invariant.
    """
    if alpha <= 0.0:
        return generated_pil
    gen = np.asarray(generated_pil.convert("RGB"), dtype=np.float32)
    ref_pil = reference_pil.convert("RGB").resize(generated_pil.size, Image.LANCZOS)
    ref = np.asarray(ref_pil, dtype=np.float32)
    h, w = gen.shape[:2]
    H = _butterworth_lowpass((h, w), cutoff=cutoff, order=order).astype(np.float32)
    H_eff = alpha * H
    out = np.empty_like(gen)
    for c in range(3):
        Fg = np.fft.fftshift(np.fft.fft2(gen[..., c]))
        Fr = np.fft.fftshift(np.fft.fft2(ref[..., c]))
        F_out = H_eff * Fr + (1.0 - H_eff) * Fg
        out[..., c] = np.fft.ifft2(np.fft.ifftshift(F_out)).real
    if mask_pil is not None:
        m = np.asarray(mask_pil.convert("L").resize(generated_pil.size, Image.NEAREST), dtype=np.float32) / 255.0
        if mask_feather > 0:
            m = cv2.GaussianBlur(m, (0, 0), float(mask_feather))
        m = m[..., None]
        out = m * gen + (1.0 - m) * out
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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
OBJECT_LABEL_PATTERN = re.compile(r"(object_\d+)")
SIZE_BUCKETS = {"small", "large"}


def sample_sort_key(item):
    sample_key, sample_name, _, _ = item
    matches = SAMPLE_NUMBER_PATTERN.findall(sample_name)
    sample_number = int(matches[-1]) if matches else float("inf")
    return sample_number, sample_key


def object_label_for_sample(sample_name):
    match = OBJECT_LABEL_PATTERN.search(sample_name)
    return match.group(1) if match else "unknown"


def size_bucket_for_sample(rel_parent):
    if not rel_parent:
        return "all"
    first = rel_parent.split(os.sep, 1)[0]
    return first if first in SIZE_BUCKETS else "all"


def evenly_spaced(items, limit):
    if limit is None or limit < 0 or len(items) <= limit:
        return list(items)
    if limit <= 0:
        return []
    if limit == 1:
        return [items[0]]

    max_idx = len(items) - 1
    selected = []
    seen = set()
    for i in range(limit):
        idx = int(round(i * max_idx / (limit - 1)))
        while idx in seen and idx < max_idx:
            idx += 1
        while idx in seen and idx > 0:
            idx -= 1
        if idx not in seen:
            selected.append(items[idx])
            seen.add(idx)
    return selected


def limit_samples_per_object(samples, max_per_object):
    if max_per_object is None or max_per_object < 0:
        return samples

    groups = {}
    for item in samples:
        _, sample_name, rel_parent, _ = item
        object_label = object_label_for_sample(sample_name)
        size_bucket = size_bucket_for_sample(rel_parent)
        groups.setdefault(object_label, {}).setdefault(size_bucket, []).append(item)

    selected_sample_names = set()
    for buckets in groups.values():
        active_buckets = [name for name, items in sorted(buckets.items()) if items]
        if not active_buckets:
            continue
        if active_buckets == ["all"]:
            for item in evenly_spaced(buckets["all"], max_per_object):
                selected_sample_names.add(item[1])
            continue

        quota = {name: min(len(buckets[name]), max_per_object // len(active_buckets)) for name in active_buckets}
        remainder = max_per_object - sum(quota.values())
        while remainder > 0:
            added = False
            for name in active_buckets:
                if quota[name] < len(buckets[name]):
                    quota[name] += 1
                    remainder -= 1
                    added = True
                    if remainder == 0:
                        break
            if not added:
                break

        for name in active_buckets:
            for item in evenly_spaced(buckets[name], quota[name]):
                selected_sample_names.add(item[1])

    return [item for item in samples if item[1] in selected_sample_names]


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


def output_paths(output_dir, rel_parent, sample, flat_output, output_ext):
    ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
    if flat_output:
        result_dir = os.path.join(output_dir, f".tmp_{sample}")
        output_path = os.path.join(output_dir, f"{sample}{ext}")
    else:
        result_dir = os.path.join(output_dir, sample)
        output_path = os.path.join(result_dir, "results_highres.png")
    return result_dir, output_path


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
    parser.add_argument('--flat_output', action='store_true', help='Save output_dir/<sample>.jpg instead of output_dir/<sample>/results_highres.png')
    parser.add_argument('--output_ext', type=str, default='.jpg')
    parser.add_argument('--default_prompt', type=str, default='an industrial object on a railway track at a steel mill')
    parser.add_argument('--skip-existing', '--skip_existing', dest='skip_existing', action='store_true', help='Skip samples whose output image already exists')
    parser.add_argument('--max-per-object', type=int, default=None, help='Process at most this many samples per object')
    parser.add_argument('--lowfreq_alpha', type=float, default=0.0, help='Strength of Butterworth low-freq replacement (0 disables, 1 = fully replace low band with original bg).')
    parser.add_argument('--lowfreq_cutoff', type=float, default=0.1, help='Butterworth low-pass cutoff as a fraction of max radial frequency (0-1). 0.1 = innermost 10%% of the spectrum.')
    parser.add_argument('--lowfreq_order', type=int, default=4, help='Butterworth filter order (higher = sharper cutoff).')
    parser.add_argument('--lowfreq_protect_fg', action='store_true', help='Use the object mask to skip low-freq replacement inside the foreground region.')
    parser.add_argument('--lowfreq_mask_feather', type=float, default=4.0, help='Feathering sigma for the protect-fg mask edge (pixels in crop space).')

    args = parser.parse_args()

    metadata_prompts = load_metadata_prompts(args.data_dir)
    all_samples = discover_samples(args.data_dir)
    samples = limit_samples_per_object(all_samples, args.max_per_object)
    if args.max_per_object is not None:
        print(f"sample_limit=max_per_object:{args.max_per_object} selected_samples={len(samples)} total_samples={len(all_samples)}")
    ext = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"
    existing_outputs = collect_existing_outputs(args.output_dir, args.flat_output, ext) if args.skip_existing else {}

    pending_samples = []
    os.makedirs(args.output_dir, exist_ok=True)
    for sample_key, sample, rel_parent, sample_dir in samples:
        result_dir, output_path = output_paths(args.output_dir, rel_parent, sample, args.flat_output, ext)
        existing_output = existing_outputs.get(sample)
        if args.skip_existing and (os.path.exists(output_path) or existing_output):
            print(f"[skip-existing] {sample_key}: {existing_output or output_path}")
            continue
        pending_samples.append((sample_key, sample, rel_parent, sample_dir, result_dir, output_path))

    if args.skip_existing:
        print(f"samples_total={len(samples)} existing_outputs={len(existing_outputs)} pending_samples={len(pending_samples)}")
    if not pending_samples:
        print("no pending samples; all outputs already exist")
        return

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

    for k, (sample_key, sample, rel_parent, sample_dir, result_dir, output_path) in enumerate(pending_samples):
        prompt = prompt_for_sample(sample, metadata_prompts, args.default_prompt)
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

        if args.lowfreq_alpha > 0.0:
            fg_mask_for_protect = cropped_mask if args.lowfreq_protect_fg else None
            restored_crop = inject_lowfreq(
                restored_crop,
                cropped_bg,
                alpha=args.lowfreq_alpha,
                cutoff=args.lowfreq_cutoff,
                order=args.lowfreq_order,
                mask_pil=fg_mask_for_protect,
                mask_feather=args.lowfreq_mask_feather,
            )

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