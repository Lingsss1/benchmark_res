# Dataset preparation and licensing

RecEvoBench does not redistribute raw or prepared third-party data. Each user
downloads data from the official source, accepts its terms, then runs the
deterministic adapter. The generated `dataset-card.json` records source and
split checksums, row counts, adapter version, cutoffs and redistribution notice.

## Amazon Reviews

Source: <https://amazon-reviews-2023.github.io/>

Accepted input is one line-delimited JSON or JSON.GZ review file. Both the 2023
field names (`user_id`, `parent_asin`, `timestamp`, `rating`) and legacy names
(`reviewerID`, `asin`, `unixReviewTime`, `overall`) are supported.

```bash
recevobench prepare --task amazon-seqrec-v0 \
  --raw /data/All_Beauty.jsonl.gz --output /data/recevobench/amazon
```

The adapter applies a 5-interaction user core, orders each user by timestamp,
uses all but the last three events for training, and reserves the last three
for public feedback, hidden selection and sealed test respectively.

## KuaiRec

Source: <https://kuairec.com/>. The upstream repository states CC-BY-SA-4.0.

The input directory must contain `big_matrix.csv` and `small_matrix.csv`.

```bash
recevobench prepare --task kuairec-watchtime-v0 \
  --raw /data/KuaiRec/data --output /data/recevobench/kuairec
```

The large natural-exposure matrix supplies train/public/hidden data. The
near-fully-observed small matrix is written only to the private sealed split.
The continuous target is exactly `play_duration / video_duration` unless the
official `watch_ratio` column is present.

## Taobao UserBehavior

Source: <https://tianchi.aliyun.com/dataset/649>. Manual agreement to Tianchi
terms may be required.

```bash
recevobench prepare --task taobao-purchase-v0 \
  --raw /data/UserBehavior.csv --output /data/recevobench/taobao
```

The adapter supports the official headerless five-column file and a named-header
variant. It maps `pv/fav/cart/buy` to `click/favorite/cart/purchase` and uses
global chronological windows. For the official time range the frozen boundaries
are 2017-12-01, 02 and 03 UTC; non-official tiny fixtures use deterministic
70/80/90 percent time-range boundaries.

## Integrity

```bash
recevobench verify-data --prepared /data/recevobench/amazon
```

Any changed row, missing file, path escape or checksum mismatch invalidates the
prepared dataset. Public agent containers receive only the `public/` directory.

