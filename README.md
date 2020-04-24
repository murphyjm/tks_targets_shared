#TKS RV Target Selection (shared)
Repository with RV target selection code for TESS candidate follow-up using Keck (and eventually atmospheric characterization with JWST).

## Notes on generating the tables from Exoplanet Archive:
Pulling new tables from exoplanet archive subject to the filters below. The goal of this table is to find known planets that could also be good targets for transmission spectroscopy. For transmission spectroscopy, the known planets need to have a reliable mass measurement and must be transiting.

### Composite table
**Add additional columns**
- Planet mass or M*sin(i) upper uncertainty
- Planet mass or M*sin(i) lower uncertainty
- Planet mass or M*sin(i) Limit Flag
- Planet Transit Flag at the bottom of Planet Parameters
- Under Stellar Columns, switch coordinates from sexagesimal to decimal degrees
Press the Update button

**Column filters**
- Planet mass or M*sin(i) upper uncertainty: not null
- Planet mass or M*sin(i) lower uncertainty: not null
These filters get rid of planets that have no upper or lower uncertainties listed on the mass measurements
- Planet mass or M*sin(i) Limit Flag: !1
Makes sure the mass reported isn’t just an upper or lower limit
- Planet mass or M*sin(i) Provenance: not like m-r
Gets rid of planets in the composite table that have a mass estimate from the M-R relationship
- Planet Transit Flag: 1
The planets must be transiting

### Confirmed table
Just use this table to get a J-magnitude, if possible, for planets that survive all of the filtering in the composite table
- Only select columns planet name and J-mag (2MASS) [mag]
If there are planets in both the filtered composite table and this list from the confirmed planet table, then we can replace the Ks mag in the IR magnitude column with a J mag. If not, just leave the K-mag there.

## Getting TIC information from ExoFOP
**Extract TIC IDs and get ExoFOP information**
Run the get_tic_ids.py script, example command line call:
  python get_tic_ids.py tic_ids
Once the files generated by get_tic_ids.py are saved:
Go to the ExoFOP-TESS Search page: https://exofop.ipac.caltech.edu/tess/search.php
- Change coordinates to Decimal Degrees (is this necessary?)
- Under the Magnitudes tab, select "Include in Output" for V, J, H, K
- Upload each .txt file and click the search button
- Save the output to the data/exofop folder
