"""
RunPod Serverless Handler for Qwen Image Editor with LoRA adapters.
"""

import base64
import io
import os
import runpod
import torch
from PIL import Image

# Global model references (lazy-loaded on first request)
pipe = None
current_lora = None

# LoRA adapter mapping — update these paths to match your HuggingFace repo structure
LORA_MAP = {
    "image_editing": "image_editing",
    "subject_driven": "subject_driven",
    "style_transfer": "style_transfer",
    "inpainting": "inpainting",
}

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # Update to your actual model ID
LORA_BASE = "your-hf-username/qwen-image-edit-loras"  # Update to your LoRA repo


def load_pipeline():
    """Load the base model pipeline once."""
    global pipe
    if pipe is not None:
        return pipe

    from diffusers import StableDiffusionImg2ImgPipeline  # adjust to actual pipeline class

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN"),
    ).to("cuda")

    return pipe


def apply_lora(lora_name: str):
    """Load/swap LoRA adapter if different from current one."""
    global current_lora, pipe

    if lora_name == current_lora:
        return

    if current_lora is not None:
        pipe.unload_lora_weights()

    if lora_name and lora_name in LORA_MAP:
        adapter_name = LORA_MAP[lora_name]
        pipe.load_lora_weights(
            LORA_BASE,
            weight_name=f"{adapter_name}.safetensors",
            token=os.environ.get("HF_TOKEN"),
        )
        current_lora = lora_name


def handler(job):
    """Process a single image editing job."""
    job_input = job["input"]

    # Parse inputs
    image_b64 = job_input["image"]
    prompt = job_input.get("prompt", "")
    lora_name = job_input.get("lora", "image_editing")
    steps = job_input.get("steps", 28)
    guidance_scale = job_input.get("guidance_scale", 3.5)
    seed = job_input.get("seed", -1)
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)

    # Decode input image
    image_bytes = base64.b64decode(image_b64)
    source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    source_image = source_image.resize((width, height))

    # Load model and LoRA
    load_pipeline()
    apply_lora(lora_name)

    # Set seed
    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Run inference
    result = pipe(
        prompt=prompt,
        image=source_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Encode result as base64
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image": result_b64}


runpod.serverless.start({"handler": handler})
