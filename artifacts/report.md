# Ovi Demo Report

This toy demo recreates one narrow mechanism from Ovi: bidirectional cross-modal attention between
audio and video streams with RoPE-based temporal positions. The audio stream runs at
`64` tokens while the video stream runs at `16` tokens. The only
change between the two variants is the audio RoPE scale.

## Result

- `unscaled` audio RoPE (`scale=1.0`) keeps the audio positions on their native faster grid.
- `scaled` audio RoPE (`scale=0.2500`) compresses the audio grid to the
  video timeline, matching Ovi's central alignment trick.

### Video -> Audio retrieval

- Unscaled accuracy: `0.250`
- Scaled accuracy: `0.938`
- Unscaled alignment error: `24.00` audio tokens
- Scaled alignment error: `1.50` audio tokens

### Audio -> Video retrieval

- Unscaled accuracy: `0.250`
- Scaled accuracy: `0.844`
- Unscaled alignment error: `5.09` video tokens
- Scaled alignment error: `0.23` video tokens

## Faithful vs simplified

- Faithful to Ovi: twin branches, bidirectional cross-attention, mismatched temporal resolutions,
  and RoPE scaling to align them.
- Omitted on purpose: diffusion / flow matching, latent audio-video VAEs, the shared T5 text
  encoder, multimillion-sample training, and any learned weights.
