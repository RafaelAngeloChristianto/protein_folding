# compbio_fp

Simple protein folding demo (simulated annealing).

Usage
------
Run the demo (interactive):

```powershell
python -m compbio_fp
```

Run with a sequence non-interactively and save animation (requires ffmpeg or ImageMagick):

```powershell
python -m scripts.run_demo --sequence ACDEFG --steps 200 --no-plot --save-animation out/demo
```

Dependencies
------------
- numpy
- matplotlib

For MP4 output install ffmpeg and ensure it's in your PATH. If ffmpeg is unavailable, the script will try ImageMagick or fallback to saving a final PNG.
