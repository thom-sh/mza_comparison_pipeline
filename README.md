# MZA Validation and Sensitivity Analysis

This repository contains the code developed for the Master’s thesis:

**“Validation and Sensitivity Analysis of an Automatic Multizone Assignment Algorithm.”**

The work focuses on validating automatically generated thermal zoning against
ground-truth floor plans and on analysing the sensitivity of MZA-related and
building parameters with respect to thermal performance indicators.

---

## Scope of the Repository

The repository includes code for:

- Preprocessing and normalisation of real and synthetic floor plans  
- Construction of geometric and topological ground truth representations  
- Quantitative comparison between externally generated MZA output and ground truth  
  (e.g. IoU, adjacency, region-level metrics)  
- Visualisation of validation results  

**Note:**  
The Multizone Assignment Algorithm (MZA) itself is executed in a separate
repository and is treated as an external input in this work.

**Out of scope:**
- Raw floor plan datasets  
- Simulation outputs and large result files  
- Thesis text and figures

## GML Input Preparation for MZA

The code for generating the CityGML input files required by the Multizone
Assignment Algorithm (MZA) is located in the `gml_msd/` folder.

This part of the workflow is responsible for preparing and exporting
building-specific GML inputs based on the MSD data structure. The generated
GML files are then used as inputs for MZA in a separate repository.

### Directory Structure and Data Exchange

All input and output directories used by the scripts in `gml_msd/` should be
configured to reference the shared data exchange folder (`austausch/`).

### Required User Modification

The only required change when processing different buildings is the
specification of the correct **building ID** corresponding to the target
building.

