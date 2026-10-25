# Module time share

Collected profiles on NVIDIA GeForce RTX 5060 Ti, 16 GB.

SageAttention, Qwen in NF4, compile torch. Each cell is one profiled generation. Lite stays resident. Pro uses block offload. The base clip is 480×864 with audio. Full HD is that clip plus the distilled 2-step model at scale 2.25. Distilled checkpoints use PiFlow, 10 steps, and guidance 1. The other checkpoints use 50 steps, guidance 5, and MagCache.

The seconds in the table are that profile, base plus super-resolution. A slice is that module's share of the same total. Slices under 0.01% are omitted.

## Charts

### Lite distill

```mermaid
pie showData
    title lite-distill-sd
    "dit" : 53.55
    "vae_decoder" : 36.25
    "text_encoder" : 5.92
    "vocoder" : 0.21
    "other" : 4.06
```

```mermaid
pie showData
    title lite-distill-fhd
    "dit" : 15.30
    "vae_decoder" : 10.35
    "text_encoder" : 1.35
    "vocoder" : 0.06
    "sr" : 71.78
    "other" : 1.15
```

### Pro distill

```mermaid
pie showData
    title pro-distill-sd
    "dit" : 85.02
    "vae_decoder" : 8.62
    "text_encoder" : 1.28
    "vocoder" : 0.06
    "other" : 5.01
```

```mermaid
pie showData
    title pro-distill-fhd
    "dit" : 52.55
    "vae_decoder" : 5.36
    "text_encoder" : 0.75
    "vocoder" : 0.03
    "sr" : 38.05
    "other" : 3.25
```

### Lite

```mermaid
pie showData
    title lite-sd
    "dit" : 89.53
    "vae_decoder" : 7.55
    "text_encoder" : 1.01
    "vocoder" : 0.04
    "other" : 1.87
```

```mermaid
pie showData
    title lite-fhd
    "dit" : 58.11
    "vae_decoder" : 4.90
    "text_encoder" : 0.68
    "vocoder" : 0.03
    "sr" : 35.12
    "other" : 1.16
```

### Pro

```mermaid
pie showData
    title pro-sd
    "dit" : 96.48
    "vae_decoder" : 1.97
    "text_encoder" : 0.30
    "vocoder" : 0.01
    "other" : 1.23
```

```mermaid
pie showData
    title pro-fhd
    "dit" : 84.69
    "vae_decoder" : 1.73
    "text_encoder" : 0.25
    "vocoder" : 0.01
    "sr" : 12.21
    "other" : 1.10
```

## Share of total time, %

| run | s | offload | dit | vae_decoder | text_encoder | vocoder | audio_vae_decoder | text_encoder_2 | sr | other |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lite-distill-sd | 173.7 | none | 53.5 | 36.2 | 5.92 | 0.21 | &lt;0.01 | &lt;0.01 | 0 | 4.06 |
| lite-distill-fhd | 608.4 | none | 15.3 | 10.4 | 1.35 | 0.06 | &lt;0.01 | &lt;0.01 | 71.8 | 1.15 |
| pro-distill-sd | 717.6 | block | 85.0 | 8.62 | 1.28 | 0.06 | &lt;0.01 | &lt;0.01 | 0 | 5.01 |
| pro-distill-fhd | 1153.0 | block | 52.6 | 5.36 | 0.75 | 0.03 | &lt;0.01 | &lt;0.01 | 38.0 | 3.25 |
| lite-sd | 834.9 | none | 89.5 | 7.55 | 1.01 | 0.04 | &lt;0.01 | &lt;0.01 | 0 | 1.87 |
| lite-fhd | 1286.7 | none | 58.1 | 4.90 | 0.68 | 0.03 | &lt;0.01 | &lt;0.01 | 35.1 | 1.16 |
| pro-sd | 3143.6 | block | 96.5 | 1.97 | 0.30 | 0.01 | &lt;0.01 | &lt;0.01 | 0 | 1.23 |
| pro-fhd | 3571.4 | block | 84.7 | 1.73 | 0.25 | 0.01 | &lt;0.01 | &lt;0.01 | 12.2 | 1.10 |
