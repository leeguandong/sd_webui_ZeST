import os
import gradio as gr
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from rembg import remove
from ip_adapter import IPAdapterXL, IPAdapter
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

import torch
import cv2
from torchvision.transforms import Compose

from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

from modules import script_callbacks

"""
Get ZeST Ready
"""

file_path = os.path.dirname(os.path.realpath(__file__))


class ZeSTInterface:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sd_ip_model = None
        self.sdxl_ip_model = None
        self.dpt_model = None
        self.init_zest_sdxl()
        self.init_zest_sd()

    def init_zest_sdxl(self):
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = os.path.join(file_path, "IP-Adapter/models/image_encoder")
        ip_ckpt = os.path.join(file_path, "IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin")
        # image_encoder_path = "models/image_encoder"
        # ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
        controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
        torch.cuda.empty_cache()

        # load SDXL pipeline
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16).to(self.device)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(self.device)
        pipe.unet = register_cross_attention_hook(pipe.unet)

        self.sdxl_ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, self.device)

    def init_zest_sd(self):
        base_model_path = "runwayml/stable-diffusion-v1-5"
        image_encoder_path = os.path.join(file_path, "IP-Adapter/models/image_encoder")
        ip_ckpt = os.path.join(file_path, "IP-Adapter/models/ip-adapter_sd15_vit-G.bin")
        # image_encoder_path = "models/image_encoder"
        # ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
        controlnet_path = "lllyasviel/sd-controlnet-depth"
        torch.cuda.empty_cache()

        # load SDXL pipeline
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16).to(self.device)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(self.device)

        pipe.unet = register_cross_attention_hook(pipe.unet)

        self.sd_ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, self.device)

    def dpt(self):
        """
        Get Depth Model Ready
        """

        model_path = os.path.join(file_path, "DPT/weights/dpt_hybrid-midas-501f0c75.pt")

        self.dpt_model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        self.dpt_model.eval()

    def run_dpt_model(self, input_image):
        """
        Compute depth map from input_image
        """
        net_w = net_h = 384
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        img = np.array(input_image)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).unsqueeze(0)

            # if optimize == True and device == torch.device("cuda"):
            #     sample = sample.to(memory_format=torch.channels_last)
            #     sample = sample.half()

            prediction = self.dpt_model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        depth_min = prediction.min()
        depth_max = prediction.max()
        bits = 2
        max_val = (2 ** (8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (prediction - depth_min) / (depth_max - depth_min)
        else:
            # out = np.zeros(prediction.shape, dtype=depth.dtype)
            out = np.zeros(prediction.shape, dtype=prediction.dtype)

        out = (out / 256).astype('uint8')
        depth_map = Image.fromarray(out).resize((1024, 1024))
        return depth_map

    def run(self, input_image, material_exemplar, sdxl=True):
        depth_map = self.run_dpt_model(input_image)

        """
        Process foreground decolored image
        """
        rm_bg = remove(input_image)
        target_mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')
        # mask_target_img = ImageChops.lighter(input_image, target_mask)
        invert_target_mask = ImageChops.invert(target_mask)

        gray_target_image = input_image.convert('L').convert('RGB')
        gray_target_image = ImageEnhance.Brightness(gray_target_image)
        factor = 1.0  # Try adjusting this to get the desired brightness
        gray_target_image = gray_target_image.enhance(factor)

        grayscale_img = ImageChops.darker(gray_target_image, target_mask)
        img_black_mask = ImageChops.darker(input_image, invert_target_mask)
        grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
        init_img = grayscale_init_img

        """
        Process material exemplar and resize all images
        """
        ip_image = material_exemplar.resize((1024, 1024))
        init_img = init_img.resize((1024, 1024))
        mask = target_mask.resize((1024, 1024))

        num_samples = 1
        if sdxl:
            images = self.sdxl_ip_model.generate(
                pil_image=ip_image,
                image=init_img,
                control_image=depth_map,
                mask_image=mask,
                controlnet_conditioning_scale=0.9,
                num_samples=num_samples,
                num_inference_steps=30,
                seed=42)
        else:
            images = self.sd_ip_model.generate(
                pil_image=ip_image,
                image=init_img,
                control_image=depth_map,
                mask_image=mask,
                controlnet_conditioning_scale=0.9,
                num_samples=num_samples,
                num_inference_steps=30,
                seed=42)

        return images[0]


interface = ZeSTInterface()


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as Zest:
        with gr.Row():
            with gr.Tab("sdxl", elem_id="sdxl_tab"):
                input_image = gr.Image(type="pil")
                input_image2 = gr.Image(type="pil")

                with gr.Row():
                    sdxl_btn = gr.Button("Run ZeST_SDXL", elem_id="ZeSTSDXL_btn", variant="primary")

            with gr.Column():
                out_gallery_kwargs = dict(columns=2, object_fit="contain", preview=True)
                output = gr.Gallery(label="output", elem_id="output", show_label=True).style(
                    **out_gallery_kwargs)

        sdxl_btn.click(interface.run,
                       inputs=[input_image, input_image2],
                       outputs=output,
                       show_progress=True)
    return [(Zest, "ZeST", "ZeST")]


script_callbacks.on_ui_tabs(on_ui_tabs)
