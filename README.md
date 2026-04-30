# data
This study constructed a comprehensive Ti-6Al-4V LPBF database covering a wide range of process-performance relationships,
containing 138 sets of processing parameters and corresponding mechanical properties from 46 papers.
The collected parameters included laser power, scanning speed, volume energy density,
powder type, particle size, layer thickness, spacing, scanning strategy, rotation angle,
and spot size; the mechanical properties included ultimate tensile strength (UTS) and overall elongation at break (TE).
The original data can be found in ./data/data_train.xlsx
# original_literature
All the original literature can be found in ./original literature
# The workflow
In each iteration, two new combinations are selected from 780 unexplored datasets.
For each selected parameter combination, three samples are fabricated and tested, and the average UTS and TE values are recorded.
After the first iteration, the parameter combination was added to the database with its fidelity parameter set to 1.0 for the next iteration.
The second iteration followed a similar process, updating the surrogate model with the updated 140 parameter combinations.

# Literature source influence analysis
The source attribution workflow ranks original literature PDFs by their leave-one-source-out influence on the final two rows of `data/data_train_2.xlsx`.
Those final two rows are treated as the second-iteration measured target points and are excluded from attribution training to avoid leakage.

Install the modeling dependencies, then run:

```bash
pip install -r requirements.txt
python -m analysis.source_influence --data data/data_train_2.xlsx --literature-dir original_literature
```

Outputs are written to `analysis_outputs/source_influence/`:

- `row_source_map.csv`
- `target_points.csv`
- `literature_influence.csv`
- `top_literature_mechanism.md`

PDF filenames are interpreted as 1-based data-row ranges, so `1.pdf` maps to row 1 and `2-8.pdf` maps to rows 2 through 8.
Only rows 1-138 are treated as original literature rows; later rows are experiment iterations.
If filename ranges have gaps or overlaps, the script stops and writes `row_source_mapping_diagnostics.csv` in the output directory.
