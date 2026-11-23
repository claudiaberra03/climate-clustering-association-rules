# Data

The input datasets are not distributed in this repository. The observations come from the ARPA Lombardia data request service, and the terms for redistributing them have not been confirmed. The synthetic series were produced with LARS-WG 8.0 (Rothamsted Research) from those observations. The saved outputs (cluster labels, frequent itemsets and rules) are included.

## Sources

- **Observations.** Daily minimum and maximum temperature and daily precipitation for 28 stations in the provinces of Milan, Como and Varese, requested from ARPA Lombardia (https://www.arpalombardia.it/temi-ambientali/meteo-e-clima/form-richiesta-dati/).
- **Synthetic series.** Produced with LARS-WG 8.0 using the global climate model CMCC-ESM2 under SSP1-2.6 and SSP5-8.5, for 2031–2050 and 2051–2070. The years 2035 and 2055 are used.

## Files expected by the notebooks

| Folder | Files | Content |
| --- | --- | --- |
| `clustering/data/` | `clustering2020.csv`, `clustering<year><scenario>.csv` (for example `clustering2035ssp126.csv`) | One row per station: `LOCALITÀ`, `LATITUDINE`, `LONGITUDINE`, then the daily minima `MIN<day>` and maxima `MAX<day>`; separator `;` |
| `association_rules/data/` | `ra_dati2020.csv`, `ra_dati<year>_<scenario>.csv` (for example `ra_dati2035_ssp126.csv`) | The same layout, restricted to the 356 days common to all years |
