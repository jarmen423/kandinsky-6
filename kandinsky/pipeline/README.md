# Image conditioning: I2AV and FLF2AV

The native pipeline accepts one or two reference images per sample. The outer
image list is always the batch; a nested pair specifies the first and last
references of one video.

```python
from kandinsky import get_pipeline

pipe = get_pipeline("kandinsky/configs/devices/h100.yaml")

# One I2AV sample; existing single-image API is preserved.
pipe(
    text="The person walks forward.", image="first.png",
    visual_cond_scheme="tail_cond_first_frame", save_path="i2av.mp4",
)

# One FLF2AV sample; two references select tail conditioning automatically.
pipe(
    text="The person walks toward the doorway.",
    image=[["first.png", "last.png"]], save_path="flf2av.mp4",
)

# Two independent I2AV samples.
pipe(
    text=["A person walks.", "A cat turns around."],
    image=["person.png", "cat.png"],
    visual_cond_scheme="tail_cond_first_frame",
    save_path=["person_i2av.mp4", "cat_i2av.mp4"],
)

# Two FLF2AV samples.
pipe(
    text=["A person walks.", "A cat turns around."],
    image=[["person_first.png", "person_last.png"], ["cat_first.png", "cat_last.png"]],
    save_path=["person_flf2av.mp4", "cat_flf2av.mp4"],
)
```

All samples must have the same number of reference images. Mixed one-/two-frame
batches, empty reference lists, more than two references, and None references
raise ValueError before image encoding. Nested single-image lists are accepted.
For one image, the existing visual_cond_scheme setting is preserved.

## Conditioning layout

For T generated latent frames, FLF2AV uses:

```text
Sequence: [generated 0 ... generated T-1, first reference, last reference]
Role IDs: [0           ... 0,             1,               1]
RoPE t:   [0           ... T-1,           0,               T-1]
Mask:     [0           ... 0,             1,               1]
```

Both references use the existing conditioning embedding. They are kept clean
during denoising and removed before VAE decoding. Video and audio duration
exclude reference tokens. The first and last output frames are generated,
not replaced with the reference images. The last reference is encoded at the
same output resolution as the first.

The two-reference path requires temporal patch_size=1 and at least two role
embeddings. Architecture, weights and training code are unchanged.

## Prompt handling and entry points

Prompt beautification currently sees only the first reference. Set
`beautifier.name: none` in the config when using captions that already describe
the transition between both frames. The nested-image API belongs to the native
Python pipeline; CLI --image still accepts one image. ComfyUI is not extended.

## Code layout

The native pipeline handles image inputs, VAE offload and batch validation.
Shared tensor transforms live in `core/algo/visual_conditioning.py`, exported
through `core/ports.py`: reference packing, role ids, decode mask, boundary RoPE
and clean-reference restoration. Euler and PiFlow use the same helpers.
No new sampler, role embedding, runtime policy or FLF2AV-specific denoise loop
is added. Existing sampler signatures and the single-image API are preserved.

# Audio and Video Conditioning

Two modes are selected automatically by the input argument:

- `audio=...`: A2V, generate video conditioned on audio.
- `video=...`: V2A, generate audio conditioned on video.
- Without either argument, existing T2AV/I2AV behavior is unchanged.

Image conditioning and audio/video conditioning cannot be combined.

## Usage

Minimal `config.yaml` for Lite; use `checkpoint: lite-distill` for PiFlow:

```yaml
checkpoint: lite
cache: {mode: none}
offload: {strategy: module}
compile: {strategy: none}
beautifier: {name: none}
```

```python
from kandinsky import get_pipeline

pipe = get_pipeline('config.yaml')
pipe(text='A person speaks.', audio='speech.wav', save_path='a2v.mp4')
pipe(text='Ocean waves and seabirds.', video='silent.mp4', save_path='v2a.mp4')
```

## Behavior and Limits

- Text remains an input in both modes.
- The supplied modality is VAE-encoded, fixed during denoising, and receives
  `t=0` on every step. The other modality starts from noise. Both towers
  interact through fused attention; this also applies to PiFlow.
- Output reuses source video frames or audio, not their VAE reconstruction.
  MP4 compression can still change pixels or audio.
- Audio becomes mono at the model sample rate. Video is sampled at the
  pipeline FPS and resized to the requested dimensions.
- Length follows `latent_frames`/`time_length`. Longer inputs are trimmed;
  shorter inputs raise `ValueError`. Audio shortages below one video frame
  are zero-padded.
- One media file per call. Prompt batches, both media inputs together,
  image/media combinations, and caching are rejected.
- A multimodal model, audio VAE and vocoder are required. For A2V, keep
  `audio_vae.need_vae_encoder=true`. Do not set `sample_audio=False`.
- GPU-tested on Lite and Lite-Distill/PiFlow; Pro and compilation are untested.

## Code layout

File decoding and device placement belong to the native pipeline (`media.py`,
`pipeline.py`). Tensor-only VAE encoding is in `core/algo/conditioning.py`,
exported through `core/ports.py` for framework ports. The helpers do not load
weights, move modules, read files, or require PyAV. Existing sampler signatures
are unchanged; the pipeline passes clean latents and the sampling flags.
