# Data acquisition and integrity boundary

All three CSVs are **excluded** from this repository. The published sources below
are for obtaining the parent collections and a public redistributed target.
**Checking an exact file SHA-256 is a prerequisite**, not a substitute for verifying
packet-to-flow extraction and label accuracy.

| Role | Required local file | Rows | SHA-256 prefix |
|---|---|---:|---|
| 2017 Friday binary source | `data/raw/ddos.csv` | 225,745 | `6ff1580f5f81` |
| 2019 redistributed target | `data/raw/cicddos2019/cicddos2019_dataset.csv` | 431,371 | `97af3b16e45f` |
| 2018 21 February target | `data/raw/cicids2018/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | `a5f4a1c2689e` |

The complete hashes and supplied class counts are in [`input_fingerprints.json`](input_fingerprints.json).

## CIC-IDS2017

Official provider: https://www.unb.ca/cic/datasets/ids-2017.html

The study input contains 225,745 rows: 128,027 DDoS and 97,718 BENIGN.
Its SHA-256 agrees with a public copy of the complete
`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` file. The study input is
named `data/raw/ddos.csv`; place the byte-identical CSV there to run the scripts.
A fresh byte-level comparison against the provider-hosted original has not been
performed. Do not infer packet-to-flow extraction or labeling validity from
this match.

## CIC-DDoS2019 redistributed target

Public **Mendeley Data Version 1**, DOI: https://doi.org/10.17632/ssnc74xm6r.1

Download `cicddos2019_dataset.csv` and place it under
`data/raw/cicddos2019/` (or keep a checked copy outside the repo and pass
its documented path to the scripts). The study downloaded the same public
version separately and confirmed the bytes match the local 431,371-row input.
**This is not the full official CIC-DDoS2019 collection** and the redistribution's
upstream extraction/sampling/label lineage remains unresolved.

## CSE-CIC-IDS2018 daily file

Official dataset page: https://www.unb.ca/cic/datasets/ids-2018.html
AWS registry: https://registry.opendata.aws/cse-cic-ids2018/

The processed daily CSVs are available through the public S3 bucket
`s3://cse-cic-ids2018/` (region `ca-central-1`). A reproducible acquisition
command for the named object to verify is:

```bash
mkdir -p data/raw/cicids2018
aws s3 cp --no-sign-request --region ca-central-1 \
  's3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv' \
  'data/raw/cicids2018/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv'
```

**Unverified:** the current S3 object's checksum has *not* been compared with
the study input in this package. The author's study-file acquisition date
was reported as 31 October 2026. The exact number of data rows is
1,048,575 (`2^20 − 1`), as also described in earlier independent studies.
This suggests a possible historical export limit but does not prove truncation.

## Check all local inputs

From repository root:

```bash
python3 tools/verify_input_fingerprints.py --root .
```

It checks local file bytes and physical line counts; it does not reconstruct
raw PCAP flows or certify labels. Do not add the CSVs to Git without
reviewing the provider's conditions and checking for personal information.
