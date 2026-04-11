# Datasets

This repository includes two public-facing dataset resources:

- `demo/`: a small runnable subset for smoke testing and README examples.
- `benchmark_prompts.csv`: the prompt manifest for the benchmark used in the paper.

What is included in `demo/`:

- sample folders in the same layout expected by the pipeline
- `prompt.txt`
- cached `processed/` assets such as masks, entity metadata, and Canny maps

What is not included:

- the full internal benchmark images and cached assets for every benchmark case
- any third-party model weights

Expected sample layout:

```text
datasets/<dataset_name>/<sample_id>/
  <one input image: png/jpg/jpeg>
  prompt.txt
  processed/
    entities.json
    masks/*.png
    canny.png
```

The public demo subset is intended for reproducibility checks and quick smoke tests.
For large-scale experiments, point `paths.dataset_root` to your full benchmark copy.
