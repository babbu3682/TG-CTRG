# Inference container (stage 6)

Thin submission image for the VLM3D report-generation task. Weights are not baked in: they are
mounted at `/weights` at runtime (download them with `scripts/hf_download_weights.sh`, see the
[root README](../README.md#6-inference-container)).

```
./build.sh                # docker build (vendored sources in src/vendor are part of the repo)
./test.sh                 # run on ./test/<case>.nii.gz with ./weights populated
./export.sh               # save the image as a tarball for submission
```

`predict.py` is the script that ran inside the submitted image.
`src/pipeline/` holds the six stages (preprocess → encoder → CLS/RET templates → CTRG-3D draws →
RadBERT labelling → label-vote ensemble + repair/verify); `src/vendor/` is the model code snapshot
the container runs; `verify/` contains the parity checks used while building the container.
