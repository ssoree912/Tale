import cv2
import torch
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from typing import Union, Tuple, Optional
from torchvision import transforms as tvt


def load_bg(
    img_path: str, 
    target_size: Union[int, Tuple[int, int]]
) -> torch.Tensor:
    bg_img = Image.open(img_path).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        bg_img = bg_img.resize(target_size, Image.Resampling.LANCZOS)    
    return tvt.ToTensor()(bg_img)[None, ...]


def load_fg(
    img_path: str, 
    mask_path: str, 
    seg_path: str,
    target_size: Union[int, Tuple[int, int]],
    latent_size: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    # Process mask
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(np.array(mask).astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST_EXACT)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, new_w, new_h = cv2.boundingRect(contours[0])
    center_x = x + new_w / 2
    center_y = y + new_h / 2
    center_row_from_top = round(center_y / target_size[1], 2)
    center_col_from_left = round(center_x / target_size[0], 2)
    aspect_ratio = new_h / new_w
    if aspect_ratio > 1:  
        scale = new_w * aspect_ratio / (target_size[0] / 2)  
        scale = new_h / (target_size[1] / 2) 
    else:  
        scale = new_w / (target_size[0] / 2) 
        scale = new_h / (aspect_ratio * (target_size[1] / 2) ) 
    scale = round(scale, 2)

    image = Image.open(img_path).convert("RGB")
    seg_map = Image.open(seg_path).convert("1")
    w, h = image.size
    aspect_ratio = h / w
    if aspect_ratio > 1:
        new_w = int(scale * (target_size[0] / 2)  / aspect_ratio)
        new_h = int(scale * (target_size[1] / 2) )
    else:
        new_w = int(scale * (target_size[0] / 2) )
        new_h = int(scale * (target_size[1] / 2)  * aspect_ratio)        
    image_resize = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    segmentation_map_resize = cv2.resize(np.array(seg_map).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    padded_segmentation_map = np.zeros((target_size[1], target_size[0]))
    start_x = (target_size[1] - segmentation_map_resize.shape[0]) // 2
    start_y = (target_size[0] - segmentation_map_resize.shape[1]) // 2
    padded_segmentation_map[start_x: start_x + segmentation_map_resize.shape[0], start_y: start_y + segmentation_map_resize.shape[1]] = segmentation_map_resize
    padded_image = Image.new("RGB", target_size)
    start_x = (target_size[0] - image_resize.width) // 2
    start_y = (target_size[1] - image_resize.height) // 2
    padded_image.paste(image_resize, (start_x, start_y))
    fg_img = tvt.ToTensor()(padded_image)[None, ...]

    seg = repeat(torch.tensor(padded_segmentation_map)[None, None, ...], '1 1 ... -> 1 4 ...')
    seg = seg[:, :, ::8, ::8]

    top_rr = 0.5*(target_size[0] - new_h)/target_size[0]  # xx% from the top
    bottom_rr = 0.5*(target_size[0] + new_h)/target_size[0]  
    left_rr = 0.5*(target_size[1] - new_w)/target_size[1]   # xx% from the left
    right_rr = 0.5*(target_size[1] + new_w)/target_size[1] 


    image_w, image_h = target_size
    latent_w, latent_h = latent_size
    latent_top_rr = int(top_rr * latent_h)  
    latent_bottom_rr = int(bottom_rr * latent_h)  
    latent_left_rr = int(left_rr * latent_w)  
    latent_right_rr = int(right_rr * latent_w) 

    latent_new_height = latent_bottom_rr - latent_top_rr
    latent_new_width = latent_right_rr - latent_left_rr
    
    latent_step_height2, remainder = divmod(latent_new_height, 2)
    latent_step_height1 = latent_step_height2 + remainder
    latent_step_width2, remainder = divmod(latent_new_width, 2)
    latent_step_width1 = latent_step_width2 + remainder

    latent_center_row_rm = int(center_row_from_top * latent_h)
    latent_center_col_rm = int(center_col_from_left * latent_w)

    latent_bbox = [max(0, int(latent_center_row_rm - latent_step_height1)), 
                   min(latent_h - 1, int(latent_center_row_rm + latent_step_height2)),
                   max(0, int(latent_center_col_rm - latent_step_width1)), 
                   min(latent_w - 1, int(latent_center_col_rm + latent_step_width2))]

    img_top_rr = int(top_rr * image_h)  
    img_bottom_rr = int(bottom_rr * image_h)  
    img_left_rr = int(left_rr * image_w)  
    img_right_rr = int(right_rr * image_w)

    img_new_height = img_bottom_rr - img_top_rr
    img_new_width = img_right_rr - img_left_rr
    
    img_step_height2, remainder = divmod(img_new_height, 2)
    img_step_height1 = img_step_height2 + remainder
    img_step_width2, remainder = divmod(img_new_width, 2)
    img_step_width1 = img_step_width2 + remainder 

    img_center_row_rm = int(center_row_from_top * image_h)
    img_center_col_rm = int(center_col_from_left * image_w)

    img_bbox = [max(0, int(img_center_row_rm - img_step_height1)), 
                min(image_h - 1, int(img_center_row_rm + img_step_height2)),
                max(0, int(img_center_col_rm - img_step_width1)), 
                min(image_w - 1, int(img_center_col_rm + img_step_width2))]
                
    ref_bbox = (latent_top_rr, latent_bottom_rr, latent_left_rr, latent_right_rr)

    return fg_img, seg, img_bbox, latent_bbox, ref_bbox