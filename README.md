# S_Pred_Paper

## Description
Run code for S Pred paper.
- accessible surface area (asa)
- intrinsically disordered region (idr)
- secondary structure (ss)

## Requirements
- python 3.6
- esm 0.3.0 (https://github.com/facebookresearch/esm)
- pytorch 1.7.1
- einops

## Usages

All three model weights file can be downloaded from ```https://drive.google.com/drive/folders/1sG0Zw1eVq07WyAL4SF-Hpb1GNh3q4bIo?usp=sharing```
The input file should be in format .a3m MSA file, or .json file like ours.
The output file will be in format .json.

(1) predict asa
```
python run_s_pred_asa.py --input_path examples/s_pred_asa.a3m --output_path s_pred_asa --conv_model_path s_pred_asa_weights.pth
```

(1) predict idr
```
python run_s_pred_idr.py --input_path examples/s_pred_idr.a3m --output_path s_pred_idr --conv_model_path s_pred_idr_weights.pth
```

(1) predict ss
```
python run_s_pred_ss.py --input_path examples/s_pred_ss.a3m --output_path s_pred_ss --conv_model_path s_pred_ss_weights.pth
```


## Reference

- Yiyu Hong, Juyong Lee, Junsu Ko. S-Pred: Protein Structural Property Prediction Using MSA Transformer.

- Roshan Rao, Jason Liu, Robert Verkuil, Joshua Meier, John F. Canny, Pieter Abbeel, Tom Sercu, Alexander Rives. MSA Transformer. bioRxiv 2021.02.12.430858; doi: https://doi.org/10.1101/2021.02.12.430858


## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


