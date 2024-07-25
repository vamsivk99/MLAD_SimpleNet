# Dataset Descriptions for Anomaly Detection

---
### MVTec AD
1. **Download and Extract**: Get the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it to the `data/mvtec` directory.
2. **Generate Metadata**: Run `python data/gen_benchmark/mvtec.py` to create the `data/mvtec/meta.json` file, which is compatible with the `DefaultAD` standard in `data/ad_dataset.py`.

```
data
├── mvtec
    ├── meta.json
    ├── bottle
        ├── train
            └── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        └── ground_truth
            ├── anomaly1
                ├── 000.png
```

### VisA
1. **Download and Extract**: Acquire the [VisA dataset](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) and extract it to `data/visa`.
2. **Generate Metadata**: Execute `python data/gen_benchmark/visa.py` to create the `data/visa/meta.json` file, formatted for compatibility with `DefaultAD` in `data/ad_dataset.py`.
