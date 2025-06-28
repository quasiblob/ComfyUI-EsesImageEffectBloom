# Eses Image Effect Bloom

![Eses Image Effect Bloom Node Screenshot](docs/image_effect_bloom.png)

> [!CAUTION]
> Before dowloading and using the contents of this repository, please read the LICENSE.txt and the disclaimer - Thank you!

## Description

The 'Eses Image Effect Bloom' is a ComfyUI custom node that provides a configurable bloom effect. It simulates the natural glow of bright light sources in a photographic image, allowing for artistic bloom effects using a GPU-accelerated PyTorch backend for real-time performance. 

💡 If you have ComfyUI installed, you don't need any extra dependencies!

 ⚠️ WARNING - **Blur effects can be (in general) quite slow, despite GPU acceleration!** 
  * When source image is large you may have to wait for a while, but image sizes like 2048x2048 should render in relatively short time.

 🧠 Don't expect any magical results, your image has to have discrete highlights, surrounded by overall darker environment, this way brighter areas can be emphasized.


## Features

* **Controllable Highlight Isolation**:
    * `low_threshold`: Sets the black point for the highlights, controlling what is considered a "bright" light source.
    * `high_threshold`: Sets the white point, allowing you to fine-tune the range of highlights included in the bloom effect.


* **Glow Controls**:
    * `blur_type`: Choose between a high-quality `gaussian` blur or a performance-friendly `box` blur for the glow.
    * `blur_radius`: Controls the size and softness of the glow, from a tight sheen to a wide, hazy aura.
    * `highlights_brightness`: A multiplier to increase the intensity of the glow *before* it's blended, creating a more powerful light emission.


* **Compositing Options**:
    * `blend_mode`: A suite of blend modes (`screen`, `add`, `overlay`, `soft_light`, `hard_light`) to control how the glow interacts with the base image.
    * `fade`: A final opacity slider to adjust the overall strength of the bloom effect.


## Requirements

* PyTorch – (you should have this if you have ComfyUI installed).


## Installation

1.  **Navigate to your ComfyUI custom nodes directory:**
    ```
    ComfyUI/custom_nodes/
    ```
2.  **Clone this repository:**
    ```
    git clone https://github.com/quasiblob/ComfyUI-EsesImageEffectBloom.git
    ```
3.  **Restart ComfyUI:**
    * After restarting, the "Eses Image Effect Bloom" node will be available in the "Eses Nodes/Image Adjustments" category.


## Folder Structure

```
ComfyUI-EsesImageEffectBloom/
├── init.py                     # Main module defining the custom node.
├── image_effect_bloom.py       # The Python file containing the node logic.
└── README.md                   # This file.
```


## Usage

* Connect an `image` tensor to the corresponding input. 
* Adjust the `low_threshold` and `high_threshold` sliders to isolate the parts of the image you want to glow. 
* Configure the `blur_radius`, `highlights_brightness`, and `blend_mode` to achieve the desired effect. 
* The node outputs the final `modified_image`, the generated `highlights_image` for diagnostics, and a passthrough of the original `image`.

🧠 Enable `Run (On Change)` from ComfyUI's toolbar to get automatic preview updates, when you modify node values. Connect both the `modified_image` and `highlights_image` to image preview nodes to see the results!


## Inputs

* **image** (`IMAGE`, *required*): The input image to apply the effect to.
* **mask** (`MASK`, *optional*): A passthrough mask. The effect is not currently maskable.


## Outputs

* **modified_image** (`IMAGE`): The final image with the bloom effect applied.
* **highlights_image** (`IMAGE`): The isolated, blurred glow layer before it's composited.
* **image** (`IMAGE`): A passthrough of the original input image.
* **mask** (`MASK`): A passthrough of the original input mask.


## Category

Eses Nodes/Image Adjustments


## Contributing

-   Feel free to report bugs and improvement ideas in issues, but I may not have time to do anything.


## License

- See LICENSE.txt


## About

-


## Update History 

* 2025.6.25 Version 1.0.0 released


## ⚠️Disclaimer⚠️

This custom node for ComfyUI is provided "as is," without warranty of any kind, express or implied. By using this node, you agree that you are solely responsible for any outcomes or issues that may arise. Use at your own risk.


## Acknowledgements

Thanks to the ComfyUI team and community for their ongoing work!
