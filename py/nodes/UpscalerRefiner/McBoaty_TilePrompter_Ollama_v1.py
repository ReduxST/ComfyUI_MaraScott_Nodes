# py/nodes/UpscalerRefiner/McBoaty_TilePrompter_Ollama_v1.py

import os
import time
import glob
import torch
import math
from types import SimpleNamespace
import comfy
import folder_paths
import random
from io import BytesIO
import base64
from ollama import Client
from pathlib import Path

from .... import root_dir, __MARASCOTT_TEMP__
from ...utils.version import VERSION
from ...utils.log import log, get_log, COLORS
from ...inc.lib.llm import MS_Llm
from ...inc.lib.cache import MS_Cache
from .inc.prompt import Node as NodePrompt
from ...inc.lib.image import MS_Image_v2 as MS_Image
from server import PromptServer
from aiohttp import web
import numpy as np
from PIL import Image
import json

class McBoaty_TilePrompter_Ollama_v1:
    @classmethod
    def INPUT_TYPES(self):
        seed = random.randint(1, 2 ** 31)
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{    
                "pipe": ("MC_PROMPTY_PIPE_IN", {"label": "McPrompty Pipe" }),
            },
            "optional": {
                "requeue": ("INT", { "label": "requeue (automatic or manual)", "default": 0, "min": 0, "max": 99999999999, "step": 1}),                
                **NodePrompt.ENTRIES,
                "ollama_prompting_enable": (["off", "on"], {"default": "off"}),
                "ollama_process_index": ("STRING", {
                    "multiline": False,
                    "default": ""  # Empty means process all tiles
                }),
                "ollama_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": """You are an expert in ekphrasis. Your task is to analyze the input image, discreetly and silently apply any requested image modifications in your imagination, then finally actually describe that updated image from your imagination using present-tense artistic language. Don't mention the original, only describe what the new one you just imagined looks like - and do it properly using the present tense like an artist's description.

The word order in your description is also VERY important, with your first sentence being dominant. Take this into account when producing your final description. For example, when describing an object, then the object itself should go first, before any of it's features.

Assume the description is for someone who can't see the image themselves, and they only understand British English. Be clear and distinct in your langage so as to not cause any confusion. Include any quoted text or titles given by the user.

Use declarative sentences, rich in adjectives and specific details. Focus on visual elements and technical aspects. Use the present tense only, to create immediacy and say what things look "like" in a good amount of useful detail. Be like an art critic!

Use Present tense: The main verbs describing the image and its elements are in the present tense (e.g., "captures," "exhibits," "is adorned," "gazing," "creates").
Use Present perfect tense: Occasionally used to imply the result of a past action that is still relevant (e.g., "lighting has been masterfully employed").
Use Passive voice: Sometimes used to focus on the subject rather than the actor (e.g., "The image is composed").
Use Participles and gerunds: Often used to describe ongoing actions or states within the image (e.g., "cascading," "reflecting," "following").
Any art styles need to be fully described. Consider "Impressionism", for example, an art style that emerged in the late 19th century, characterized by a focus on capturing the fleeting effects of light and colour in a way that evokes an immediate, sensory impression. Rather than aiming for highly detailed or realistic depictions, Impressionist artists used loose brushwork, vibrant colours, and often short, visible strokes to convey the atmosphere or "impression" of a scene.

Example:
Your secret image is a photo of a chair and you have been asked "describe this as if it were an impressionist style oil painting". In the case of this example, your answer could be something like this:

"This impressionist style oil painting shows a simple chair, which seems to almost blend into its surroundings, thanks to it's soft, blurred edges. The focus is more on how the light falls across it, creating subtle shifts in colour. The sunlight is streaming through a nearby window, casting warm, golden highlights on one side, while cooler, shadowy lavender tones linger on the other. The brushstrokes are loose and visible, almost as if the painter captured the scene in a hurry, wanting to express the way the chair itself felt in that exact moment."

Remember, the above is an EXAMPLE only - don't just copy it verbatim as not everything is impressionism!!! Use words appropriate to the art style or relevant transformation which has been imagined."""
                }),
                "ollama_query": ("STRING", {
                    "multiline": True,
                    "default": "Describe the image."
                }),
                "ollama_debug": (["enable", "disable"], {"default": "disable"}),
                "ollama_url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "ollama_model": ("STRING", {
                    "multiline": False,
                    "default": "llava"
                }),
                "ollama_keep_alive": ("INT", {
                    "default": 5,
                    "min": -1,
                    "max": 60,
                    "step": 1
                }),
                "ollama_seed": ("INT", {
                    "default": seed,
                    "min": 0,
                    "max": 2 ** 31,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = (
        "MC_PROMPTY_PIPE_OUT",
    )
    
    RETURN_NAMES = (
        "McPrompty Pipe",
    )
    
    OUTPUT_IS_LIST = (
        False,
    )
        
    OUTPUT_NODE = True
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "A \"Tile Prompt Editor\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):
                        
        input_prompts, input_tiles = kwargs.get('pipe', (None, None))
        input_denoises = ('', ) * len(input_prompts)

        self.init(**kwargs)
        
        # Early check for Ollama enable flag
        ollama_enabled = kwargs.get('ollama_prompting_enable') == 'on'
        
        log("McBoaty (PromptEditor) is starting to do its magic", None, None, f"Node {self.INFO.id}")
        
        _input_prompts = MS_Cache.get(self.CACHE.prompt, input_prompts)
        _input_prompts_edited = MS_Cache.get(self.CACHE.prompt_edited, input_prompts)
        _input_denoises = MS_Cache.get(self.CACHE.denoise, input_denoises)
        _input_denoises_edited = MS_Cache.get(self.CACHE.denoise_edited, input_denoises)
        
        refresh = False
        
        if not MS_Cache.isset(self.CACHE.denoise):
            _input_denoises = input_denoises
            MS_Cache.set(self.CACHE.denoise, _input_denoises)
        if not MS_Cache.isset(self.CACHE.prompt) or _input_prompts != input_prompts:
            _input_prompts = input_prompts
            MS_Cache.set(self.CACHE.prompt, _input_prompts)
            _input_denoises = input_denoises
            MS_Cache.set(self.CACHE.denoise, input_denoises)
            refresh = True

        if not MS_Cache.isset(self.CACHE.denoise_edited) or refresh:
            _input_denoises_edited = input_denoises
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)
        if not MS_Cache.isset(self.CACHE.prompt_edited) or refresh:
            _input_prompts_edited = input_prompts
            MS_Cache.set(self.CACHE.prompt_edited, _input_prompts_edited)
            _input_denoises_edited = input_denoises
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)
        elif len(_input_prompts_edited) != len(_input_prompts):
            _input_prompts_edited = [gp if gp is not None else default_gp for gp, default_gp in zip(_input_prompts_edited, input_prompts)]
            MS_Cache.set(self.CACHE.prompt_edited, _input_prompts_edited)
            _input_denoises_edited = [gp if gp is not None else default_gp for gp, default_gp in zip(_input_denoises_edited, input_denoises)]
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)

        if _input_denoises_edited != _input_denoises:
            input_denoises = _input_denoises_edited
        if _input_prompts_edited != _input_prompts:
            input_prompts = _input_prompts_edited

        output_prompts_js = input_prompts
        input_prompts_js = _input_prompts
        output_prompts = output_prompts_js
        output_denoises_js = input_denoises
        input_denoises_js = _input_denoises
        output_denoises = output_denoises_js

        results = list()
        filename_prefix = "McBoaty" + "_temp_" + "tilePrompter" + "_id_" + self.INFO.id
        search_pattern = os.path.join(__MARASCOTT_TEMP__, filename_prefix + '*')
        files_to_delete = glob.glob(search_pattern)
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                # log(f"Deleted: {file_path}", None, None, f"Node {self.INFO.id} - SUCCESS")
            except Exception as e:
                log(f"Error deleting {file_path}: {e}", None, None, "Node {self.INFO.id} - ERROR")        
            
        # Parse the ollama_process_index input
        ollama_process_index = kwargs.get('ollama_process_index', '').strip()
        if ollama_process_index:
            try:
                # Remove surrounding brackets if present
                stripped = ollama_process_index.strip().strip('[]')
                # Split by comma and convert to integers
                indices = [int(idx.strip()) - 1 for idx in stripped.split(',') if idx.strip().isdigit()]
                if not indices:
                    raise ValueError("No valid indices found.")
            except Exception as e:
                log(f"Error parsing ollama_process_index: {e}. Expected format like '1, 2, 3, 24' or '[1, 2, 3, 24]'. Processing all tiles.", None, None, f"Node {self.INFO.id}")
                indices = list(range(len(input_tiles)))
        else:
            # Process all tiles if no indices are specified
            indices = list(range(len(input_tiles)))

        for index, tile in enumerate(input_tiles):
            full_output_folder, filename, counter, subfolder, subfolder_filename_prefix = folder_paths.get_save_image_path(f"MaraScott/{filename_prefix}", self.output_dir, tile.shape[1], tile.shape[0])
            file = f"{filename}_{index:05}.png"
            file_path = os.path.join(full_output_folder, file)
            
            # Save all tiles regardless of whether they'll be processed
            if not os.path.exists(file_path):
                i = 255. * tile.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)                

            # Then process with Ollama if enabled and tile is in indices
            if ollama_enabled and index in indices:
                # Get keep_alive value and adjust if needed
                keep_alive = kwargs.get('ollama_keep_alive')
                if keep_alive == 0:
                    if index < len(input_tiles) - 1:
                        keep_alive = 0.1
                
                # Use the actual file path that we just saved
                tile_path = Path(file_path)  # Convert to Path object
                
                # Get debug setting from kwargs
                debug = kwargs.get('ollama_debug')
                if debug == "enable":
                    log(f"Processing tile file: {tile_path}", None, None, f"Node {self.INFO.id}")
                    
                ollama_response = self.process_ollama_vision(
                    tile_image=tile,
                    tile_index=index,
                    tile_path=tile_path,  # Pass the actual file path
                    query=kwargs.get('ollama_query'),
                    system_prompt=kwargs.get('ollama_system_prompt'),
                    debug=debug,  # Pass the debug setting
                    url=kwargs.get('ollama_url'),
                    model=kwargs.get('ollama_model'),
                    seed=kwargs.get('ollama_seed'),
                    keep_alive=keep_alive
                )

                if ollama_response:
                    # Update output prompts
                    output_prompts = list(output_prompts)
                    output_prompts[index] = ollama_response
                    output_prompts = tuple(output_prompts)
                    output_prompts_js = output_prompts
                    
                    # Mark as edited in cache (same way manual edits are handled)
                    cache_name = f'input_prompts_{self.INFO.id}'
                    cache_name_edited = f'{cache_name}_edited'
                    _input_prompts_edited = list(MS_Cache.get(cache_name_edited, output_prompts))
                    _input_prompts_edited[index] = ollama_response
                    MS_Cache.set(cache_name_edited, tuple(_input_prompts_edited))
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "temp"
            })
            counter += 1

        log("McBoaty (PromptEditor) is done with its magic", None, None, f"Node {self.INFO.id}")
                    
        return {"ui": {
            "prompts_out": output_prompts_js, 
            "prompts_in": input_prompts_js , 
            "denoises_out": output_denoises_js, 
            "denoises_in": input_denoises_js , 
            "tiles": results,
        }, "result": ((output_prompts, output_denoises),)}

    @classmethod
    def init(self, **kwargs):
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', 0),
        )
        self.CACHE = SimpleNamespace(
            prompt = f'input_prompts_{self.INFO.id}',
            prompt_edited = None,
            denoise = f'input_denoises_{self.INFO.id}',
            denoise_edited = None,
        )
        self.CACHE.prompt_edited = f'{self.CACHE.prompt}_edited'
        self.CACHE.denoise_edited = f'{self.CACHE.denoise}_edited'
        
        self.output_dir = folder_paths.get_temp_directory()
        
    @classmethod
    def process_ollama_vision(self, tile_image, tile_index, tile_path, query, system_prompt, debug, url, model, seed, keep_alive):
        try:
            if debug == "enable":
                log(f"[Ollama Vision] Starting processing tile {tile_index}:", None, None, f"Node {self.INFO.id}")
                log(f"  Model: {model}", None, None, f"Node {self.INFO.id}")
                log(f"  URL: {url}", None, None, f"Node {self.INFO.id}")
                log(f"  Keep Alive: {keep_alive}m", None, None, f"Node {self.INFO.id}")
                log(f"  Seed: {seed}", None, None, f"Node {self.INFO.id}")

            # Create Ollama client and generate response
            client = Client(host=url)
            
            if debug == "enable":
                log(f"[Ollama Vision] Sending request:", None, None, f"Node {self.INFO.id}")
                log(f"  System prompt length: {len(system_prompt)} chars", None, None, f"Node {self.INFO.id}")
                log(f"  Query: {query}", None, None, f"Node {self.INFO.id}")

            start_time = time.time()
            response = client.chat(
                model=model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': query,
                        'images': [tile_path]
                    }
                ],
                options={
                    'seed': seed,
                },
                keep_alive=f"{keep_alive}m"
            )
            end_time = time.time()

            if debug == "enable":
                log(f"[Ollama Vision] Response received:", None, None, f"Node {self.INFO.id}")
                log(f"  Processing time: {end_time - start_time:.2f}s", None, None, f"Node {self.INFO.id}")
                log(f"  Response length: {len(response['message']['content'])} chars", None, None, f"Node {self.INFO.id}")

            return response['message']['content']
        except Exception as e:
            log(f"[Ollama Vision] Error: {str(e)}", None, None, f"Node {self.INFO.id}")
            if debug == "enable":
                import traceback
                log(f"  Traceback: {traceback.format_exc()}", None, None, f"Node {self.INFO.id}")
            return None

@PromptServer.instance.routes.get("/MaraScott/McBoaty/Ollama/v1/get_input_prompts")
async def get_input_prompts(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_prompts_{nodeId}'
    input_prompts = MS_Cache.get(cache_name, [])
    return web.json_response({ "prompts_in": input_prompts })
    
@PromptServer.instance.routes.get("/MaraScott/McBoaty/Ollama/v1/get_input_denoises")
async def get_input_denoises(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    input_denoises = MS_Cache.get(cache_name, [])
    return web.json_response({ "denoises_in": input_denoises })
    
@PromptServer.instance.routes.get("/MaraScott/McBoaty/Ollama/v1/set_prompt")
async def set_prompt(request):
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    prompt = request.query.get("prompt", None)
    
    # Debug logging
    print(f"[McBoaty] Received set_prompt - node: {nodeId}, index: {index}, prompt: {prompt[:20] if prompt else None}")
    
    cache_name = f'input_prompts_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    
    _input_prompts = MS_Cache.get(cache_name, [])
    _input_prompts_edited = MS_Cache.get(cache_name_edited, _input_prompts)
    
    if index >= 0 and index < len(_input_prompts_edited):
        # Log before update
        print(f"[McBoaty] Before update - Cache {cache_name_edited}: {_input_prompts_edited[index][:50]}")
        
        _input_prompts_edited = list(_input_prompts_edited)
        _input_prompts_edited[index] = prompt
        MS_Cache.set(cache_name_edited, tuple(_input_prompts_edited))
        
        # Log after update
        print(f"[McBoaty] After update - Cache {cache_name_edited}: {_input_prompts_edited[index][:50]}")
    else:
        print(f"[McBoaty] Invalid index {index} for node {nodeId}")
    
    return web.json_response(f"Tile {index} prompt has been updated :{prompt}")

@PromptServer.instance.routes.get("/MaraScott/McBoaty/Ollama/v1/set_denoise")
async def set_denoise(request):
    denoise = request.query.get("denoise", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_denoises = MS_Cache.get(cache_name, [])
    _input_denoises_edited = MS_Cache.get(cache_name_edited, _input_denoises)
    if _input_denoises_edited and index < len(_input_denoises_edited):
        _input_denoises_edited_list = list(_input_denoises_edited)
        _input_denoises_edited_list[index] = denoise
        _input_denoises_edited = tuple(_input_denoises_edited_list)
        MS_Cache.set(cache_name_edited, _input_denoises_edited)
    return web.json_response(f"Tile {index} denoise has been updated: {denoise}")

@PromptServer.instance.routes.get("/MaraScott/McBoaty/Ollama/v1/tile_prompt")
async def tile_prompt(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    type = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = os.path.join(root_dir, type)
    image_path = os.path.abspath(os.path.join(
        target_dir, 
        request.query.get("subfolder", ""), 
        request.query["filename"]
    ))
    c = os.path.commonpath((image_path, target_dir))
    if c != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    return web.json_response(f"here is the prompt \n{image_path}")