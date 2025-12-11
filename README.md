<div align=center>
<h1>TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation</h1>

![GitHub Repo stars](https://img.shields.io/github/stars/yehjin-shin/TV-Rec)
 [![arXiv](https://img.shields.io/badge/arXiv-2510.25259-b31b1b.svg)](https://arxiv.org/abs/2510.25259)

<div>
    <a href="https://yehjin-shin.github.io/" target="_blank"><b>Yehjin Shin</b></a><sup>1</sup>,
      <a href="https://www.jeongwhanchoi.com" target="_blank"><b>Jeongwhan Choi</b></a><sup>2*</sup>,
      <a href="https://scholar.google.com/citations?user=4GpvarsAAAAJ&hl=en" target="_blank">Seojin Kim</a><sup>1</sup>,
      <a href="https://sites.google.com/view/noseong" target="_blank">Noseong Park</a><sup>3</sup>,
    <div>
     <sup>1</sup>KAIST
    </div>
</div>
</div>

---

This is the official PyTorch implementation of the NeurIPS 2025 paper "TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation".

## 📚 Related Works from Our Group

| Venue | Paper | Code |
|:-----:|:------|:----:|
| **NeurIPS 2024** | [Graph Convolutions Enrich the Self-attention in Transformers!](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5eceb48c3bc8b5d936c05ff8e2ece65e-Abstract-Conference.html) | [![GitHub](https://img.shields.io/github/stars/jeongwhanchoi/gfsa?style=social)](https://github.com/jeongwhanchoi/gfsa) |
| **ICML 2024** | [Polynomial-based Self-Attention for Table Representation Learning](https://proceedings.mlr.press/v235/kim24ae.html) | [![GitHub](https://img.shields.io/github/stars/JayoungKim408/TabPSA?style=social)](https://github.com/JayoungKim408/TabPSA) |
| **AAAI 2024** | [An Attentive Inductive Bias for Sequential Recommendation Beyond the Self-Attention](https://ojs.aaai.org/index.php/AAAI/article/view/28747) | [![GitHub](https://img.shields.io/github/stars/yehjin-shin/BSARec?style=social)](https://github.com/yehjin-shin/BSARec) |
| **IJCAI 2025** | [Learning Advanced Self-Attention for Linear Transformers in the Singular Value Domain](https://www.ijcai.org/proceedings/2025/0730) | [![GitHub](https://img.shields.io/github/stars/hyowonwi/agf?style=social)](https://github.com/hyowonwi/agf) |

---

## 1. Install conda environments 

```
conda env create -f tvrec_env.yaml
conda activate tvrec
```

## 2. Train TVRec
Note that pretrained model (.pt) and train log file (.log) will saved in `TVRec/output`
### (1) How to train
- `train_name`: name for log file and checkpoint file
```
python main.py  --data_name=[DATASET] \
                --lr=[LEARNING_RATE] \
                --M=[M] \ 
                --hidden_dropout_prob=[DROPOUT_RATE] \
                --reg_weight=[REGULARIZER_WEIGHT] \
                --train_name=[LOG_NAME]
```
### (2) Example: Beauty
```
python main.py  --data_name=Beauty \
                --lr=5e-4 \
                --M=32 \ 
                --hidden_dropout_prob=0.5 \
                --reg_weight=0 \
                --train_name=TVRec_Beauty_best
```
### (3) Example: LastFM
```
python main.py  --data_name LastFM \
                --lr=1e-3 \
                 --M=8 \
                 --hidden_dropout_prob=0.4 \
                 --reg_weight=1e-3 \
                 --train_name=TVRec_LastFM_best
```

## 3. Test pretrained TVRec
Note that pretrained model (.pt file) must be in `TVRec/output`
### (1) How to test pretrained model
- `load_model`: pretrained model name without .pt
```
python main.py  --data_name [DATASET] \
                --M=[M] \ 
                --hidden_dropout_prob [DROPOUT_RATE] \
                --reg_weight [REGULARIZER_WEIGHT] \
                --load_model [PRETRAINED_MODEL_NAME] \
                --do_eval
```
### (2) Beauty
```
python main.py  --data_name Beauty \
                --M=32 \ 
                --load_model TVRec_Beauty_best \
                --do_eval
```
### (3) LastFM
```
python main.py  --data_name LastFM \
                --M=8 \ 
                --load_model TVRec_LastFM_best \
                --do_eval
```


## Citation
If you find our work useful, please cite:
```bibtex
@inproceedings{shin2025tvrec,
  title={{TV}-Rec: Time-Variant Convolutional Filter for Sequential Recommendation},
  author={Yehjin Shin and Jeongwhan Choi and Seojin Kim and Noseong Park},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=cWEssTIwG5}
}
```

---

<div align="center">
<p>If you have any questions, please open an issue or contact us at <a href="mailto:yehjin.shin@kaist.ac.kr">yehjin.shin@kaist.ac.kr</a>
<p>⭐ Star this repository if you find it helpful!</p>
</div>
