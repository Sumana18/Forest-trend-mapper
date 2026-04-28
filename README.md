---
title: Forest Trend Mapper
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8765
license: mit
---

# Forest Trend Mapper
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19860349.svg)](https://doi.org/10.5281/zenodo.19860349)

This is an interactive web application built using Python (Solara), Geemap, and Google Earth Engine. It is designed to visualize vegetation trends and the impact of climate and fire history on vegetation dynamics across the Interior Alaska boreal forest. 

🚀 **[Access the Live Application Here](https://sumanasahoo-forest-trend-mapper.hf.space/)**

📄 **[Read the Accompanying Research Paper (MDPI Forests)](https://www.mdpi.com/1999-4907/16/5/777)**

## Key Features
* **Interactive Spatial Analysis:** Draw a Region of Interest (ROI) directly on the map, set custom year ranges, and calculate vegetation trends such as the Kendall τ and Slope (rates of change).
* **Comprehensive Data Integration:** The tool can access multiple robust environmental data sources, including TerraClimate for meteorological data, MTBS (Monitoring Trends in Burn Severity) for US fire perimeters, and AICC Alaska Fire History shapefiles.
* **Time-Series Visualization:** By clicking on specific points on the map, users can generate time-series plots that track spectral indices (such as NDVI, NBR, and NDMI) and correlate them with climate variables (like Summer Max Temperature).
* **Data Export:** Researchers can easily download point-specific time-series data as a CSV for further offline analysis.

## Citation
If you use this application or find the related research helpful, please cite as follows:

**Paper:**
> Sahoo, S., et al. (2025). Interplay of Topography, Fire History, and Climate on Interior Alaska Boreal Forest Vegetation Dynamics. *Forests*, 16(5), 777. https://doi.org/10.3390/f16050777

**Software:**
> Sahoo, S. (2026). Forest Trend Mapper (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.19860349
