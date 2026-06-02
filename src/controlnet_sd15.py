import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from utils.paths import HF_CACHE_ROOT, resolve_model_identifier
from src.sd15 import SCHED_MAP


class SD15ControlNetPipeline:
    def __init__(
        self,
        base_model_path: str,
        controlnet_model_path: str,
        scheduler_name: str = "euler_a",
        dtype: str = "float16",
        enable_vae_slicing: bool = True,
        enable_xformers: bool = True,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("CUDA GPU is required.")

        torch_dtype = torch.float16 if str(dtype).lower() == "float16" else torch.float32

        resolved_base = resolve_model_identifier(base_model_path)
        resolved_controlnet = resolve_model_identifier(controlnet_model_path)

        controlnet = ControlNetModel.from_pretrained(
            resolved_controlnet,
            torch_dtype=torch_dtype,
            cache_dir=str(HF_CACHE_ROOT),
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            resolved_base,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            cache_dir=str(HF_CACHE_ROOT),
        )

        if scheduler_name not in SCHED_MAP:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        self.pipe.scheduler = SCHED_MAP[scheduler_name].from_config(self.pipe.scheduler.config)

        self.pipe = self.pipe.to(device)

        if enable_vae_slicing:
            self.pipe.enable_vae_slicing()

        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.device = device
        self.dtype = torch_dtype

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        conditioning_image,
        conditioning_scale: float,
        guess_mode: bool,
        generator,
        step_callback=None,
        controlnet_end_ratio: float = 1.0,
    ):
        controlnet_end_ratio = max(0.0, min(1.0, float(controlnet_end_ratio)))
        if controlnet_end_ratio <= 0.0:
            # diffusers requires start < end; when ratio=0 we treat it as "disabled"
            # by zeroing the conditioning scale and nudging the end ratio slightly above 0.
            conditioning_scale = 0.0
            controlnet_end_ratio = 1e-3
        callback = None
        callback_steps = 1
        if step_callback is not None:
            def _cb(step, timestep, latents):
                step_callback(step, int(timestep), latents)

            callback = _cb
            callback_steps = 1

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=conditioning_image,
            controlnet_conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=callback,
            callback_steps=callback_steps,
            control_guidance_start=0.0,
            control_guidance_end=controlnet_end_ratio,
        )
        image = out.images[0]
        extra = {"nsfw": getattr(out, "nsfw_content_detected", None)}
        return image, extra
