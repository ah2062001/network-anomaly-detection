# Table regeneration

Run `python3 table_reproduction/rebuild_all_ten_tables.py` from the publication
folder. The script produces ten LaTeX tabular fragments in
`table_reproduction/regenerated_tables/` from archived CSV/JSON files in
`table_reproduction/evidence_for_tables/`. It requires Python 3 standard
library only and does not retrain any model.

To verify the generated tables against your Overleaf project, run:

```bash
python3 table_reproduction/rebuild_all_ten_tables.py --overleaf /path/to/Overleaf_Project
```
