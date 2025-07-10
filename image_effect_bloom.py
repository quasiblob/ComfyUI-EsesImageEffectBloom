# ==========================================================================
# Eses Image Effects Bloom
# ==========================================================================
#
# Description:
# The 'Eses Image Effects Bloom' node creates a post-processing effect that
# simulates the glow of bright light sources in an image. It works by
# isolating the brightest parts of the image, blurring them, and then
# blending them back over the original image for a soft, luminous effect.
# The entire process is GPU-accelerated using PyTorch.
#
# Key Features:
#
# - Highlight Isolation:
#   - low_threshold: Sets the black point for the highlights.
#   - high_threshold: Sets the white point, controlling the highlight range.
#
# - Glow Controls:
#   - effect_type: Choose between a high-quality 'gaussian', a fast 'box' blur,
#     or a cinematic 'light_streaks' effect.
#   - blur_radius: Controls the size and softness of the glow.
#   - highlights_brightness: Multiplies the brightness of the glow before blending.
#
# - Compositing:
#   - blend_mode: A wide variety of blend modes (e.g., screen, add, overlay)
#     to control how the glow interacts with the original image.
#   - fade: Adjusts the final intensity of the bloom effect.
#
# Usage:
# Connect an image to the 'image' input. The primary output is 'modified_image'.
# 'highlights_image' shows the isolated glow layer for diagnostics, and
# 'image' outputs the original, unaltered image for convenience.
#
# Version: 1.1.1
#
# License: see LICENSE.txt
#
# ==========================================================================

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np



# =================================================================================
# Color Space and Blur Helpers for PyTorch Tensors
# =================================================================================


# Blur effects --------------

def _gaussian_blur_separable(image: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Applies a fast, separable Gaussian blur.
    """
    sigma = max(radius, 0.01) # Ensure sigma is not zero
    kernel_size = int(sigma * 4) * 2 + 1
    
    kernel_1d_range = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-kernel_1d_range**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    channels = image.shape[1]
    
    kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    
    blurred_h = F.conv2d(image, kernel_h, padding='same', groups=channels)
    blurred_hv = F.conv2d(blurred_h, kernel_v, padding='same', groups=channels)
    
    return blurred_hv

def _streaks_blur(image: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Applies a directional blur to create a 'streaks' or 'cross' effect.
    It blurs horizontally and vertically in separate passes and adds them.
    """
    sigma = max(radius, 0.01) # Ensure sigma is not zero
    kernel_size = int(sigma * 4) * 2 + 1
    
    # Create a 1D Gaussian kernel
    kernel_1d_range = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-kernel_1d_range**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    channels = image.shape[1]
    
    # Create horizontal and vertical kernels
    kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    
    # Apply horizontal and vertical blurs independently to the original image
    blurred_h = F.conv2d(image, kernel_h, padding='same', groups=channels)
    blurred_v = F.conv2d(image, kernel_v, padding='same', groups=channels)
    
    # Add the two blurred images together to create the cross effect
    return blurred_h + blurred_v

def _box_blur(image: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Applies a fast box blur. This is extremely fast.
    """
    # Kernel size must be an odd integer for symmetric padding to work correctly.
    kernel_size = int(radius * 2) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 1:
        return image

    return F.avg_pool2d(image, kernel_size=(kernel_size, kernel_size), stride=1, padding=kernel_size // 2)


# Color conversion ----------

def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Converts a B, C, H, W RGB tensor to a B, C, H, W HSV tensor."""
    epsilon = 1e-6
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    
    hue = torch.zeros_like(cmax)
    mask = cmax_idx == 0
    hue[mask] = (((rgb[:, 1:2] - rgb[:, 2:3]) / (delta + epsilon)) % 6)[mask]
    mask = cmax_idx == 1
    hue[mask] = (((rgb[:, 2:3] - rgb[:, 0:1]) / (delta + epsilon)) + 2)[mask]
    mask = cmax_idx == 2
    hue[mask] = (((rgb[:, 0:1] - rgb[:, 1:2]) / (delta + epsilon)) + 4)[mask]
    
    hue[delta == 0] = 0
    hue = hue / 6.0
    saturation = torch.where(cmax == 0, torch.zeros_like(cmax), delta / (cmax + epsilon))
    value = cmax
    
    return torch.cat([hue, saturation, value], dim=1)

def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Converts a B, C, H, W HSV tensor to a B, C, H, W RGB tensor."""
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c
    
    rgb_prime = torch.zeros_like(hsv)
    
    mask = (h >= 0) & (h < 1/6)
    values = torch.cat([c, x, torch.zeros_like(c)], dim=1)
    rgb_prime = torch.where(mask.expand_as(rgb_prime), values, rgb_prime)
    
    mask = (h >= 1/6) & (h < 2/6)
    values = torch.cat([x, c, torch.zeros_like(c)], dim=1)
    rgb_prime = torch.where(mask.expand_as(rgb_prime), values, rgb_prime)

    mask = (h >= 2/6) & (h < 3/6)
    values = torch.cat([torch.zeros_like(c), c, x], dim=1)
    rgb_prime = torch.where(mask.expand_as(rgb_prime), values, rgb_prime)

    mask = (h >= 3/6) & (h < 4/6)
    values = torch.cat([torch.zeros_like(c), x, c], dim=1)
    rgb_prime = torch.where(mask.expand_as(rgb_prime), values, rgb_prime)

    mask = (h >= 4/6) & (h < 5/6)
    values = torch.cat([x, torch.zeros_like(c), c], dim=1)
    rgb_prime = torch.where(mask.expand_as(rgb_prime), values, rgb_prime)

    mask = (h >= 5/6) & (h <= 1)
    values = torch.cat([c, torch.zeros_like(c), x], dim=1)
    rgb_prime = torch.where(mask.expand_as(rgb_prime), values, rgb_prime)
    
    return rgb_prime + m



# =================================================================================
# Main Node Class
# =================================================================================

class EsesImageEffectBloom:
    NODE_NAME = "Eses Image Effect Bloom"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("modified_image", "highlights_image", "image", "mask",)
    FUNCTION = "apply_bloom"
    CATEGORY = "Eses Nodes/Image Adjustments"

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.001, # MODIFIED: Increased precision
                    "tooltip": "Pixels darker than this value will not glow."
                }),
                "high_threshold": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.001, # MODIFIED: Increased precision
                    "tooltip": "Sets the white point for the highlights, controlling the range of pixels that glow."
                }),
                "effect_type": (["gaussian", "box", "light_streaks"], {
                    "default": "gaussian",
                    "tooltip": "'gaussian' is high-quality, 'box' is fast, 'light_streaks' creates a cross-like glow."
                }),
                "blur_radius": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 512.0, "step": 0.1,
                    "tooltip": "Controls the size and softness of the glow in pixels."
                }),
                "highlights_brightness": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01, # MODIFIED: Increased max range
                    "tooltip": "A multiplier for the glow's intensity before it's blended with the image."
                }),
                "blend_mode": (["screen", "add", "multiply", "overlay", "soft_light", "hard_light"], {
                    "tooltip": "The blend mode used to composite the glow onto the original image."
                }),
                "fade": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Adjusts the final overall intensity of the bloom effect. 0.0 is no effect, 1.0 is full effect."
                }),
                "blur_resolution_limit_px": ("INT", {
                    "default": 2048, "min": 512, "max": 4096, "step": 64,
                    "tooltip": "Sets the max resolution for the blur calculation to manage performance. Lower values (e.g., 512) are much faster but may reduce quality for small blurs."
                }),
            }, "optional": { "mask": ("MASK",) }
        }

    def apply_bloom(self, image: torch.Tensor, low_threshold: float, high_threshold: float, effect_type: str, blur_radius: float, highlights_brightness: float, blend_mode: str, fade: float, blur_resolution_limit_px: int, mask: torch.Tensor = None):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input 'image' must be a torch.Tensor")

        base = image.clone()
        
        if high_threshold <= low_threshold:
            high_threshold = low_threshold + 1e-6

        # --- Step 1: Isolate Highlights ---
        image_bchw = base.permute(0, 3, 1, 2)
        hsv_image = rgb_to_hsv(image_bchw)
        h, s, v = hsv_image[:, 0:1], hsv_image[:, 1:2], hsv_image[:, 2:3]
        v_modified = torch.clamp((v - low_threshold) / (high_threshold - low_threshold), 0, 1)
        highlights_hsv = torch.cat([h, s, v_modified], dim=1)
        highlights_image = hsv_to_rgb(highlights_hsv)
        
        blurred_highlights = highlights_image

        if blur_radius > 0:
            # --- Step 2: Blur The Highlights (with optimizations) ---

            # --- A. Get original image dimensions ---
            _b, _c, original_h, original_w = highlights_image.shape
            original_long_side = float(max(original_h, original_w))
            
            # --- B. Cap initial processing resolution using the user-defined limit ---
            processing_long_side = min(original_long_side, float(blur_resolution_limit_px))
            image_to_blur = highlights_image

            if original_long_side > processing_long_side:
                scale_factor = processing_long_side / original_long_side
                target_h = int(original_h * scale_factor)
                target_w = int(original_w * scale_factor)
                image_to_blur = F.interpolate(image_to_blur, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # --- C. For large blur radii, dynamically downscale the image even further ---
            MIN_BLUR_FOR_SCALING = 32.0
            MAX_BLUR_FOR_SCALING = 512.0
            DIM_AT_MAX_BLUR = 256.0

            if blur_radius > MIN_BLUR_FOR_SCALING:
                t = (min(blur_radius, MAX_BLUR_FOR_SCALING) - MIN_BLUR_FOR_SCALING) / (MAX_BLUR_FOR_SCALING - MIN_BLUR_FOR_SCALING)
                
                current_h, current_w = image_to_blur.shape[2], image_to_blur.shape[3]
                current_long_side = float(max(current_h, current_w))
                
                target_long_side = torch.lerp(torch.tensor(current_long_side), torch.tensor(DIM_AT_MAX_BLUR), t).item()

                if target_long_side < current_long_side:
                    dynamic_scale_factor = target_long_side / current_long_side
                    target_h = int(current_h * dynamic_scale_factor)
                    target_w = int(current_w * dynamic_scale_factor)
                    image_to_blur = F.interpolate(image_to_blur, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # --- D. Calculate the effective blur radius based on the total downscaling ---
            final_h, final_w = image_to_blur.shape[2], image_to_blur.shape[3]
            total_scale_factor = max(final_h, final_w) / original_long_side
            effective_blur_radius = blur_radius * total_scale_factor

            # --- E. Apply the blur to the (potentially tiny) image ---
            if effect_type == 'gaussian':
                blurred_highlights_small = _gaussian_blur_separable(image_to_blur, effective_blur_radius)
            elif effect_type == 'light_streaks':
                blurred_highlights_small = _streaks_blur(image_to_blur, effective_blur_radius)
            else: # box
                blurred_highlights_small = _box_blur(image_to_blur, effective_blur_radius)

            # --- F. Upscale the final blurred result back to the original image size ---
            blurred_highlights = F.interpolate(blurred_highlights_small, size=(original_h, original_w), mode='bicubic', align_corners=False)

        blend = blurred_highlights.permute(0, 2, 3, 1)
        
        # --- Step 3: Apply Highlights Brightness ---
        if highlights_brightness != 1.0:
            blend = torch.clamp(blend * highlights_brightness, 0, 1)
        
        # --- Step 4: Composite Over Original ---
        if blend_mode == 'screen':
            composited_image = 1.0 - (1.0 - base) * (1.0 - blend)
        elif blend_mode == 'add':
            composited_image = base + blend
        elif blend_mode == 'multiply':
            composited_image = base * blend
        elif blend_mode == 'overlay':
            composited_image = torch.where(base <= 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
        elif blend_mode == 'hard_light':
            composited_image = torch.where(blend <= 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
        elif blend_mode == 'soft_light':
            composited_image = torch.where(blend <= 0.5, base - (1 - 2 * blend) * base * (1 - base), base + (2 * blend - 1) * (torch.sqrt(base.clamp(min=0)) - base))
        else: # Fallback
            composited_image = 1.0 - (1.0 - base) * (1.0 - blend)
    
        composited_image = torch.clamp(composited_image, 0, 1)

        # --- Step 5: Fade the Effect ---
        modified_image = torch.lerp(base, composited_image, fade)

        # --- Final Output ---
        if mask is None:
            mask = torch.ones(1, image.shape[1], image.shape[2], dtype=torch.float32, device=image.device)
            
        return (modified_image, blend, base, mask,)
    
