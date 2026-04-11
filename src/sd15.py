import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
)

from utils.paths import HF_CACHE_ROOT, resolve_model_identifier

SCHED_MAP = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
    "dpmsolver": DPMSolverMultistepScheduler,
    "heun": HeunDiscreteScheduler,
}


class SD15Pipeline:
    def __init__(self, model_path, scheduler_name="euler_a", dtype="float16",
                 enable_vae_slicing=True, enable_xformers=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("CUDA GPU is required.")

        dtype = torch.float16 if str(dtype).lower() == "float16" else torch.float32

        # Resolve local snapshot or model ID consistently and reuse cache directory.
        resolved_model_path = resolve_model_identifier(model_path)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            resolved_model_path,
            torch_dtype=dtype,
            cache_dir=str(HF_CACHE_ROOT),
        )

        # Validate scheduler name early so config issues fail fast.
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
                # xFormers is optional. Missing package only affects memory usage.
                pass

        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def generate(self, prompt, negative_prompt, width, height, steps, guidance_scale, generator, step_callback=None):
        """
        step_callback(step_idx:int, timestep:int, latents:torch.Tensor) -> None
        This follows diffusers callback(step, timestep, latents).
        """
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
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=callback,
            callback_steps=callback_steps,
        )
        image = out.images[0]
        extra = {"nsfw": getattr(out, "nsfw_content_detected", None)}
        return image, extra
