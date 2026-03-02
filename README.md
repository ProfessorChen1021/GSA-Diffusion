# GSA-Diffusion 🚀

<p align="center">
  <img src="https://readme-typing-svg.demolab.com/?font=JetBrains+Mono&size=22&duration=2800&pause=800&color=7CF8FF&center=true&vCenter=true&width=620&lines=Grounded+SAM+%2B+ADI+%2B+ControlNet;Entity-aware+text-to-image;Plug+your+LLM+planner+or+rule+mode;Reproducible+runs+with+cached+masks" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pipeline-Grounded--SAM_%2B_ControlNet_%2B_ADI-8A2BE2?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/badge/Models-SD1.5_%7C_Canny_ControlNet-2196F3?style=for-the-badge&logo=codeproject" />
  <img src="https://img.shields.io/badge/Planner-LLM_%7C_Rule-43A047?style=for-the-badge&logo=semantic-web" />
  <img src="https://img.shields.io/badge/Outputs-Reproducible-26C6DA?style=for-the-badge&logo=hack-the-box" />
  <img src="https://img.shields.io/badge/License-Apache--2.0-FF7043?style=for-the-badge&logo=apache" />
</p>

> Neon-fast, mask-aware SD1.5 generation: LLM/rule planner → Grounding DINO → SAM → Canny ControlNet → ADI attention injection.

## ⚡ What’s inside
- Grounded masks: Grounding DINO finds boxes, SAM carves pixel-perfect masks.
- ADI injection: per-entity spatial gating + self-attention isolation with mask-aware cross-attn.
- ControlNet Canny: configurable guidance schedule (`controlnet_ratio`).
- Planner-aware: rule or your own LLM API (`src/llm_client.py` hook).
- Reproducible: cached `processed/`, timestamped runs, per-seed folders with meta/config snapshots.

## 🎛️ Run in 60 seconds
```bash
pip install -r requirements.txt                # torch CUDA build recommended
# drop data like: datasets/000001/{image.png,prompt.txt}
python run_inference.py                        # uses config/default.yaml
```

Outputs land in `outputs/<NsamplesMseeds_YYMMDD_HHMM>/<sample_id>/seed_<seed>/` with `result.png`, `canny.png`, `meta.json`, `time.txt`, `gpu_mem_gb.txt`.

## 🛰️ Pipeline in 4 moves
1) Planner → `prompt.txt` → entities (`phrase_entity`, `phrase_ent_attr`) → cache `processed/entities.json`.  
2) Grounded-SAM → DINO boxes → SAM masks → `processed/masks/*.png`.  
3) Canny → `processed/canny.png` for ControlNet.  
4) Sampling → ADI context hooks UNet attn → SD1.5+ControlNet across all seeds.

## 🗺️ Config hotspots (`config/default.yaml`)
- `paths.base_model_path` / `controlnet.model_path` / `dataset_root` / `output_root`
- `params`: `width/height`, `num_inference_steps`, `guidance_scale`, `scheduler`, `dtype`, `seeds`, `max_samples`
- `controlnet`: `conditioning_scale`, `guess_mode`, `canny_low_threshold`, `canny_high_threshold`
- `planner.mode`: `rule` | `llm` (LLM uses `src/llm_client.call_llm_api`)
- `injection`: `controlnet_ratio`, `adi_ratio`, `adi_linear_decay`
- `adi`: `alpha_g`, `alpha_e`, `sharpen_t`, `enable_global`
- `method`: pick ADI/baseline via dotted path + `method_kwargs`

## 🧭 Layout
```
GSA-Diffusion/
├─ config/            # default.yaml
├─ datasets/          # 000001/{image.png,prompt.txt,processed/}
├─ outputs/           # runs live here
├─ src/               # pipeline, planner, grounder, controlnet wrappers
├─ methods/           # ADI attention logic & baselines
├─ utils/             # IO, image, seed, timers, config helpers
└─ run_inference.py   # entry point
```

## 🔬 Dev tips
- Delete a sample’s `processed/` to force re-run of planner/masks; Canny will auto-regen if missing.
- Tweak `max_samples` to select ranges: `null`=all, `5`=first 5, `[13,78]`=inclusive slice, `[0,0]`=all.
- Attention maps auto-save at 0.4T/0.7T when `attn_records` is enabled (see `methods/adi_attention.py`).

## 📜 License
Apache-2.0. Check model licenses (SD1.5, ControlNet, Grounding DINO, SAM) for your use case.
