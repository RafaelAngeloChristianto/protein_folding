# compbio_fp

Simple protein folding demo (simulated annealing) with AlphaFold comparison.

Usage
------
Run the demo (interactive):

```powershell
python -m compbio_fp
```

Run with a sequence non-interactively and save animation (requires ffmpeg or ImageMagick):

```powershell
python -m compbio_fp --sequence ACDEFG --steps 200 --no-plot --save-animation out/demo
```

Run with AlphaFold comparison:

```powershell
python -m compbio_fp --sequence MRWQEMGYIFYPRKlr --uniprot A0A0C5B5G6
```

Launch GUI for easy sequence selection and comparison:

```powershell
python gui_main.py
```

Dependencies
------------
- numpy
- matplotlib

For MP4 output install ffmpeg and ensure it's in your PATH. If ffmpeg is unavailable, the script will try ImageMagick or fallback to saving a final PNG.
