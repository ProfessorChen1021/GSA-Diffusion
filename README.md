# GSA-Diffusion

<p align="center">
  <img src="https://readme-typing-svg.demolab.com/?font=JetBrains+Mono&size=22&duration=2800&pause=800&color=7CF8FF&center=true&vCenter=true&width=620&lines=Grounded+SAM+%2B+ADI+%2B+ControlNet;Entity-aware+text-to-image;Plug+your+LLM+planner+or+rule+mode;Reproducible+runs+with+cached+masks" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pipeline-Grounded--SAM_%2B_ControlNet_%2B_ADI-8A2BE2?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/badge/Models-SD1.5_%7C_SDXL-2196F3?style=for-the-badge&logo=codeproject" />
  <img src="https://img.shields.io/badge/Planner-LLM_%7C_Rule-43A047?style=for-the-badge&logo=semantic-web" />
  <img src="https://img.shields.io/badge/Outputs-Reproducible-26C6DA?style=for-the-badge&logo=hack-the-box" />
  <img src="https://img.shields.io/badge/License-Apache--2.0-FF7043?style=for-the-badge&logo=apache" />
</p>

> Mask-aware SD generation: LLM or rule planner -> GroundingDINO -> SAM -> Canny ControlNet -> ADI attention injection.

## What's inside
- Grounded masks: GroundingDINO finds boxes, and SAM turns them into usable masks.
- ADI injection: per-entity spatial gating with self-attention isolation and mask-aware cross-attention edits.
- ControlNet Canny: configurable structure guidance through `controlnet_ratio`.
- Planner-aware: rule mode or your own LLM API through `src/llm_client.py`.
- Reproducible outputs: cached `processed/` assets, timestamped runs, and per-seed metadata snapshots.

## Public release setup
```bash
pip install -r requirements.txt                # torch CUDA build recommended
# public demo subset:
# datasets/demo/
# public benchmark list:
# datasets/benchmark_prompts.csv
python run_in_SD1.5.py
```

This GitHub version is prepared for paper submission and public release. It now ships:
- a small runnable demo subset in `datasets/demo/`
- the benchmark prompt manifest in `datasets/benchmark_prompts.csv`

If sample folders are present, the pipeline uses them directly. If only the CSV manifest is present, both entry scripts print a clear public-release note instead of trying to launch a full run.

Included in the public release:
- full code
- `datasets/demo/`
- `datasets/benchmark_prompts.csv`
- configs, evaluation scripts, and pipeline logic

Not included in the public release:
- the full internal benchmark assets for every test case
- private or unpublished reference data beyond the demo subset

## Full internal run
If you have the original private benchmark assets, the expected sample layout is still:

```text
datasets/<dataset_name>/<sample_id>/
  <one input image: png/jpg/jpeg>
  prompt.txt
  processed/
    entities.json
    masks/*.png
    canny.png
```

Then you can run:

```bash
python run_in_SD1.5.py
python run_in_SDXL.py --config config/sdxl.yaml
```

Outputs land in `outputs/<NsamplesMseeds_YYMMDD_HHMM>/<dataset>/<sample_id>/seed_<seed>/` with `result.png`, `canny.png`, `meta.json`, `time.txt`, and `gpu_mem_gb.txt`.

## Pipeline in 4 moves
1. Planner -> `prompt.txt` -> entities (`phrase_entity`, `phrase_ent_attr`) -> cache `processed/entities.json`.
2. Grounded-SAM -> DINO boxes -> SAM masks -> `processed/masks/*.png`.
3. Canny -> `processed/canny.png` for ControlNet.
4. Sampling -> ADI hooks UNet attention -> SD1.5 or SDXL generation across all seeds.

## Config hotspots
- `paths.base_model_path` / `controlnet.model_path` / `dataset_root` / `output_root`
- `params`: `width/height`, `num_inference_steps`, `guidance_scale`, `scheduler`, `dtype`, `seeds`, `max_samples`
- `data`: `mode`, `prompt_manifest`, and the CSV column names used in the public release
- `controlnet`: `conditioning_scale`, `guess_mode`, `canny_low_threshold`, `canny_high_threshold`
- `planner.mode`: `rule` | `llm` (LLM uses `src/llm_client.call_llm_api`)
- `injection`: `controlnet_ratio`, `adi_ratio`, `adi_linear_decay`
- `adi`: `alpha_g`, `alpha_e`, `sharpen_t`, `enable_global`
- `method`: pick ADI/baseline via dotted path + `method_kwargs`

The two config files now default to the current public-release behavior:
- `config/default.yaml`: SD1.5 config with `data.mode: "auto"`
- `config/sdxl.yaml`: SDXL config with the same public prompt manifest entry

## Layout
```text
GSA-Diffusion/
|-- config/            # default.yaml + sdxl.yaml
|-- datasets/          # demo subset + benchmark_prompts.csv
|-- src/               # pipeline, planner, grounder, controlnet wrappers
|-- methods/           # ADI attention logic and baselines
|-- utils/             # IO, image, seed, timers, config helpers
|-- run_in_SD1.5.py    # SD1.5 entry
`-- run_in_SDXL.py     # SDXL entry
```

## Evaluation
- `eval.py` keeps the masked CLIP evaluation flow.
- `VQA_evl.py` keeps the VQA-style attribute-binding checks.
- These scripts are included in the public release, but they still need generated results and masks. The prompt CSV alone is not enough to run them end to end.

## Dev tips
- Delete a sample's `processed/` folder to force a fresh planner/mask pass; Canny will regenerate if missing.
- Tweak `max_samples` to select ranges: `null` = all, `5` = first 5, `[13,78]` = inclusive slice, `[0,0]` = all.
- Attention maps auto-save at 0.4T and 0.7T when `attn_records` is enabled (see `methods/adi_attention.py`).

## License
This repository is released under Apache-2.0. See `LICENSE` for the full text.
Third-party pretrained models and any separately licensed assets keep their own original licenses and usage terms.
