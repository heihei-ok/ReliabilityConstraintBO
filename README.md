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
