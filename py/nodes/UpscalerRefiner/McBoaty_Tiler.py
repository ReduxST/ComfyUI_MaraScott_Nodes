import os
import time
import torch
import math
from types import SimpleNamespace

import comfy
import comfy_extras
import comfy_extras.nodes_custom_sampler
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
from comfy_extras.nodes_canny import Canny
import nodes
from server import PromptServer
from aiohttp import web
import folder_paths

from PIL import Image
import numpy as np

from .... import root_dir, __MARASCOTT_TEMP__
from ...utils.version import VERSION
from ...utils.log import log, get_log, COLORS
from ...inc.lib.image import MS_Image_v2 as MS_Image
from ...vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch as ColorMatch
from ...inc.lib.llm import MS_Llm
from ...inc.lib.cache import MS_Cache

from .inc.prompt import Node as NodePrompt

class McBoaty_Tiler():

    UPSCALE_METHODS = [
        "area", 
        "bicubic", 
        "bilinear", 
        "bislerp",
        "lanczos",
        "nearest-exact"
    ]
    
    COLOR_MATCH_METHODS = [   
        'none',
        'mkl',
        'hm', 
        'reinhard', 
        'mvgd', 
        'hm-mvgd-hm', 
        'hm-mkl-hm',
    ]
    
    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    KSAMPLERS = {}
    LLM = {}
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "image": ("IMAGE", {"label": "Original Image (Low-Res)" }),
                "upscaled_image": ("IMAGE", {"label": "Pre-Upscaled Image" }),
                "tile_selection": ("STRING", {"label": "Tiles to use from original (e.g. 1,3,8)", "default": ""}),
                "model": ("MODEL", { "label": "Model" }),
                "clip": ("CLIP", { "label": "Clip" }),
                "vae": ("VAE", { "label": "VAE" }),
                "positive": ("CONDITIONING", { "label": "Positive" }),
                "negative": ("CONDITIONING", { "label": "Negative" }),
                "seed": ("INT", { "label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),

                "upscale_model": (folder_paths.get_filename_list("upscale_models"), { "label": "Upscale Model" }),
                "output_upscale_method": (self.UPSCALE_METHODS, { "label": "Custom Output Upscale Method", "default": "bicubic"}),

                "tile_size": ("INT", { "label": "Tile Size", "default": 512, "min": 320, "max": 4096, "step": 64}),
                "feather_mask": ("INT", { "label": "Feather Mask", "default": 64, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "vae_encode": ("BOOLEAN", { "label": "VAE Encode type", "default": True, "label_on": "tiled", "label_off": "standard"}),
                "tile_size_vae": ("INT", { "label": "Tile Size (VAE)", "default": 512, "min": 256, "max": 4096, "step": 64}),

                "color_match_method": (self.COLOR_MATCH_METHODS, { "label": "Color Match Method", "default": 'none'}),
                "tile_prompting_active": ("BOOLEAN", { "label": "Tile prompting (with WD14 Tagger - experimental)", "default": False, "label_on": "Active", "label_off": "Inactive"}),
                "vision_llm_model": (MS_Llm.VISION_LLM_MODELS, { "label": "Vision LLM Model", "default": "microsoft/Florence-2-large" }),
                "llm_model": (MS_Llm.LLM_MODELS, { "label": "LLM Model", "default": "llama3-70b-8192" }),

            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "MC_BOATY_PIPE", 
        "MC_PROMPTY_PIPE_IN",
        "STRING",
    )
    
    RETURN_NAMES = (
        "McBoaty Pipe",
        "McPrompty Pipe",
        "info", 
    )
    
    OUTPUT_IS_LIST = (
        False,
        False,
        False,
    )
    
    
    OUTPUT_NODE = True
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "A \"TILER\" Node that combines tiles from an original image and its pre-upscaled version"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):

        start_time = time.time()
        
        self.init(**kwargs)
        
        if self.INPUTS.image is None:
            raise ValueError(f"MaraScottUpscalerRefinerNode id {self.INFO.id}: No image provided")

        if not isinstance(self.INPUTS.image, torch.Tensor):
            raise ValueError(f"MaraScottUpscalerRefinerNode id {self.INFO.id}: Image provided is not a Tensor")
        
        log("McBoaty (Upscaler) is starting to do its magic", None, None, f"Node {self.INFO.id}")
        
        self.OUTPUTS.image, image_width, image_height, image_divisible_by_8 = MS_Image().format_2_divby8(self.INPUTS.image)

        self.PARAMS.grid_specs, self.OUTPUTS.grid_images, self.OUTPUTS.grid_prompts = self.upscale(self.OUTPUTS.image, "Upscaling")

        end_time = time.time()

        output_info = self._get_info(
            image_width, 
            image_height, 
            image_divisible_by_8, 
            self.OUTPUTS.grid_prompts,
            int(end_time - start_time)
        )
        
        log("McBoaty (Upscaler) is done with its magic", None, None, f"Node {self.INFO.id}")

        output_tiles = torch.cat(self.OUTPUTS.grid_images)

        return (
            (
                self.INPUTS,
                self.PARAMS,
                self.KSAMPLER,
                self.OUTPUTS,
            ),
            (
                self.OUTPUTS.grid_prompts,
                output_tiles,
            ),
            output_info
        )
        
    @classmethod
    def init(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', None),
        )
        
        self.INPUTS = SimpleNamespace(
            image = kwargs.get('image', None),
            upscaled_image = kwargs.get('upscaled_image', None),
            tile_selection = kwargs.get('tile_selection', ""),
        )
        
        self.LLM = SimpleNamespace(
            vision_model = kwargs.get('vision_llm_model', None),
            model = kwargs.get('llm_model', None),
        )
        
        self.PARAMS = SimpleNamespace(
            upscale_model_name = kwargs.get('upscale_model', None),
            upscale_method = kwargs.get('output_upscale_method', "lanczos"),
            feather_mask = kwargs.get('feather_mask', None),
            color_match_method = kwargs.get('color_match_method', 'none'),
            upscale_size_type = None,
            upscale_size = None,
            tile_prompting_active = kwargs.get('tile_prompting_active', False),
            grid_spec = None,
            rows_qty = 1,
            cols_qty = 1,
        )
        self.PARAMS.upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(self.PARAMS.upscale_model_name)[0]

        self.KSAMPLER = SimpleNamespace(
            tiled = kwargs.get('vae_encode', None),
            tile_size = kwargs.get('tile_size', None),
            tile_size_vae = kwargs.get('tile_size_vae', None),
            model = kwargs.get('model', None),
            clip = kwargs.get('clip', None),
            vae = kwargs.get('vae', None),
            noise_seed = kwargs.get('seed', None),
            sampler_name = None,
            scheduler = None,
            positive = kwargs.get('positive', None),
            negative = kwargs.get('negative', None),
            add_noise = True,
            sigmas_type = None,
            model_type = None,
            steps = None,
            cfg = None,
            denoise = None,
            control_net_name = None,
            control = None,
        )

        self.OUTPUTS = SimpleNamespace(
            grid_images = [],
            grid_prompts = [],
            output_info = ["No info"],
            grid_tiles_to_process = [],
        )
    
        
    @classmethod
    def _get_info(self, image_width, image_height, image_divisible_by_8, output_prompts, execution_duration):
        formatted_prompts = "\n".join(f"        [{index+1}] {prompt}" for index, prompt in enumerate(output_prompts))
        
        return [f"""

    IMAGE (INPUT)
        width   :   {image_width}
        height  :   {image_height}
        image divisible by 8 : {image_divisible_by_8}

    ------------------------------

    ------------------------------
    
    TILES PROMPTS
{formatted_prompts}    
        
    ------------------------------

    EXECUTION
        DURATION : {execution_duration} seconds

    NODE INFO
        version : {VERSION}

"""]        
    
    @classmethod
    def upscale(self, image, iteration):
        
        feather_mask = self.PARAMS.feather_mask
        rows_qty_float = (image.shape[1] * self.PARAMS.upscale_model.scale) / self.KSAMPLER.tile_size
        cols_qty_float = (image.shape[2] * self.PARAMS.upscale_model.scale) / self.KSAMPLER.tile_size
        rows_qty = math.ceil(rows_qty_float)
        cols_qty = math.ceil(cols_qty_float)

        tiles_qty = rows_qty * cols_qty        
        if tiles_qty > NodePrompt.INPUT_QTY:
            msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than {NodePrompt.INPUT_QTY} ({tiles_qty} for {self.PARAMS.cols_qty} cols and {self.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {self.INFO.id} - McBoaty_Tiler")
            raise ValueError(msg)

        # Upscale the original image
        upscaled_original = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, image)[0]

        # Get the pre-upscaled image
        upscaled_image = self.INPUTS.upscaled_image

        # Verify dimensions match
        if upscaled_original.shape != upscaled_image.shape:
            msg = get_log(f"\n\n--------------------\n\nError: Pre-upscaled image dimensions ({upscaled_image.shape}) do not match upscaled original dimensions ({upscaled_original.shape}).\nPlease ensure the correct upscale model and pre-upscaled image are used.\n\n--------------------\n", "RED", "YELLOW", f"Node {self.INFO.id} - McBoaty_Tiler")
            raise ValueError(msg)

        self.PARAMS.rows_qty = rows_qty
        self.PARAMS.cols_qty = cols_qty
        
        # Get grid specifications and split both images into tiles
        grid_specs = MS_Image().get_tiled_grid_specs(upscaled_image, self.KSAMPLER.tile_size, self.PARAMS.rows_qty, self.PARAMS.cols_qty, feather_mask)[0]
        grid_images_original = MS_Image().get_grid_images(upscaled_original, grid_specs)
        grid_images_preupscaled = MS_Image().get_grid_images(upscaled_image, grid_specs)

        # Parse tile selection string
        selected_tiles = []
        if self.INPUTS.tile_selection.strip():
            try:
                # Remove brackets and split by commas
                tile_str = self.INPUTS.tile_selection.strip('[]').replace(' ', '')
                selected_tiles = [int(x) - 1 for x in tile_str.split(',') if x]  # Convert to 0-based indexing
                # Validate tile indices
                if any(i < 0 or i >= len(grid_images_original) for i in selected_tiles):
                    raise ValueError("Tile indices out of range")
            except Exception as e:
                msg = get_log(f"\n\n--------------------\n\nError parsing tile selection: {str(e)}\nFormat should be like [1,3,8]\n\n--------------------\n", "RED", "YELLOW", f"Node {self.INFO.id} - McBoaty_Tiler")
                raise ValueError(msg)

        # Combine tiles from both sources
        grid_images = []
        for i in range(len(grid_images_preupscaled)):
            if i in selected_tiles:
                grid_images.append(grid_images_original[i])
            else:
                grid_images.append(grid_images_preupscaled[i])
        
        # Generate prompts
        grid_prompts = []
        llm = MS_Llm(self.LLM.vision_model, self.LLM.model)
        prompt_context = llm.vision_llm.generate_prompt(image)
        total = len(grid_images)
        for index, grid_image in enumerate(grid_images):
            prompt_tile = prompt_context
            if self.PARAMS.tile_prompting_active:
                log(f"tile {index + 1}/{total} - [tile prompt]", None, None, f"Node {self.INFO.id} - Prompting {iteration}")
                prompt_tile = llm.generate_tile_prompt(grid_image, prompt_context, self.KSAMPLER.noise_seed)
            source = "original" if index in selected_tiles else "pre-upscaled"
            log(f"tile {index + 1}/{total} - [{source}] - [tile prompt] {prompt_tile}", None, None, f"Node {self.INFO.id} - Prompting {iteration}")
            grid_prompts.append(prompt_tile)
                            
        return grid_specs, grid_images, grid_prompts