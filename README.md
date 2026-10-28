# MAV Lab

A browser-based motion amplification workspace using the **actual MagNet model and pretrained weights** referenced in [the source Colab notebook](https://colab.research.google.com/drive/1Vn2gfWp7UJfAPzhU6yKYQBWXVD8OMJac).

## Use

1. Upload an MP4, MOV, WebM, or other browser-decodable video (up to 500 MB).
2. Select one, two, or three amplification factors. Defaults match the notebook: **10, 25, 50**. Custom factors: 1–100.
3. Choose processing resolution and run analysis. A cancellable background worker reports frame progress.
4. Play the original and each output, start all videos together, reset them, use fullscreen, or download any video.

Videos remain available in the current tab until another valid video is uploaded. Successful reruns replace generated outputs; cancellation and failed reruns preserve previous outputs. Closing/reloading the tab clears the session. No video is sent to a server or stored remotely.

Recent Chrome or Edge recommended. WebCodecs decoding/encoding and OffscreenCanvas are required. MP4/H.264 outputs are generated if the browser supports encoding, otherwise VP9/WebM. Audio is omitted. Unsupported files produce an error rather than synthetic results.

## Model, weights, resources

- Architecture: **MagNet**, encoder → learned motion manipulator → decoder, **877,968 parameters**.
- Source: [aerraj/motion_magnification_learning-based](https://github.com/aerraj/motion_magnification_learning-based), commit `be33c252f670bcb55d1442e30491584f9564a22d`. `vendor/magnet.py` retains the model implementation; its unused CUDA/data import was removed. MIT license retained in `vendor/LICENSE`.
- Original checkpoint: [magnet_epoch12_loss7.28e-02.pth](https://github.com/ZhengPeng7/motion_magnification_learning-based/releases/download/v1.0/magnet_epoch12_loss7.28e-02.pth), epoch 12, approx. 3.4 MB.
- Checkpoint SHA-256: `26c0e3ba528152d9219698ca4dac8a1924e3c4528f81277c3296fb7c034d4fae`.
- Original paper: [Learning-based Video Motion Magnification](https://arxiv.org/abs/1804.02684). This is ZhengPeng7's unofficial PyTorch implementation used by the notebook, not the paper authors' TensorFlow implementation.
- Browser graphs: `public/models/encoder.onnx` and `render.onnx`, ONNX opset 17. Weights are included; no external inference service, Python server, CUDA installation, or API key needed.
- Conversion validation: six comparisons (64×96 and 96×128, factors 10/25/50) against original PyTorch evaluation. Maximum absolute tensor difference **1.61e-6** for the committed export. Hashes and validation result in `public/models/manifest.json`.
- Runtime: ONNX Runtime Web/WASM in a dedicated worker; Mediabunny and WebCodecs decode/encode frames. The encoder is shared between factors to avoid redundant computations.

RGB frames are normalized to `[-1,1]` in NCHW layout. Like `make_frameACB.py`, inference pairs each frame with its preceding frame, then uses the current frame's texture and amplified learned motion difference. The original first frame starts each output. Output values are clipped and converted to 8-bit RGB.

Changes from notebook video plumbing: browser codecs replace FFmpeg/OpenCV; videos retain source frame timing rather than forcing 30 fps; user selects a 128–512 px maximum edge for practical browser processing. Dimensions are rounded to multiples of four (minor aspect-ratio rounding). Original upload is always unmodified. The model has **no temporal band-pass filter**. Large motion, lighting changes, camera shake, and high factors can create artifacts. This tool is for visual inspection, not a calibrated vibration measurement or automatic fault diagnosis.

Processing cost increases strongly with resolution, frame count, and number of factors. Start with a short clip at 128 or 256 px. The model runs on the device CPU; no claim of real-time performance. Model/runtime assets are fetched on first use; encoded output files accumulate in memory until replacement.

## Development

```sh
npm ci
npm run dev
npm run build
```

Vercel recognizes the Vite project. `vercel.json` specifies build and output directory. Deploy with `npx vercel --prod` or import the GitHub repository into Vercel. No environment variables needed.

To reproduce the ONNX export:

```sh
python -m venv .venv
. .venv/bin/activate
pip install torch onnx onnxruntime numpy
curl -fL https://github.com/ZhengPeng7/motion_magnification_learning-based/releases/download/v1.0/magnet_epoch12_loss7.28e-02.pth -o /tmp/magnet.pth
python scripts/export_model.py /tmp/magnet.pth
```

The export script checks numerical equivalence and generates the model provenance manifest. Included ONNX files are already ready to use.
