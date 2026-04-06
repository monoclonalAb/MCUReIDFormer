# MCUReIDFormer

MCU-constrained Neural Architecture Search for Vision Transformer-based animal person re-identification.

---

## Datasets

Each dataset must follow the flat-directory format expected by `lib/datasets.py`:

```
{dataset_root}/
  train/     {pid}_{camid}_{trackid}_{frameid}.jpg
  query/     {pid}_{camid}_{trackid}_{frameid}.jpg
  gallery/   {pid}_{camid}_{trackid}_{frameid}.jpg
```

| Dataset | Train imgs | Query | Gallery | IDs (train) | Species | Source |
|---------|-----------|-------|---------|-------------|---------|--------|
| [ATRW](https://lila.science/datasets/atrw/) | 3,730 | 424 | 521 | 149 | Amur tiger | LILA BC |
| [FriesianCattle2017](https://data.bris.ac.uk/data/dataset/2yizcfbkuv4352pzc32n54371r) | 752 | 85 | 97 | 66 | Holstein-Friesian cattle | Univ. Bristol |
| [iPanda-50](https://github.com/iPandaDateset/iPanda-50) | 5,620 | 577 | 677 | 40 | Giant panda | GitHub |
| [Lion](https://github.com/tvanzyl/wildlife_reidentification) | 594 | 61 | 85 | 75 | African lion | GitHub |
| [MPDD](https://data.mendeley.com/datasets/v5j6m8dzhv/1) | 1,032 | 104 | 521 | 95 | Dog | Mendeley |
| [SeaStarReID2023](https://lila.science/sea-star-re-id-2023/) | 1,768 | 189 | 230 | 81 | Sea star | LILA BC |
| [Stoat](https://github.com/ywu840/IndivAID) | 183 | 13 | 13 | 56 | Stoat | GitHub |
| [CoBRA](https://zenodo.org/records/7322820) | 8,578 | 1,145 | 1,715 | 32 | Pig / koi / pigeon | Zenodo |
| [PolarBearVidID](https://zenodo.org/records/7564529) | — | 5,135 | — | — | Polar bear | Zenodo |

Pass the dataset root to the training script via `--data-path`:

```bash
--data-path /path/to/ATRW --data-set ATRW
```
