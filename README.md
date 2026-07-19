# End-to-End Mapping of Microwave Data to Synthetic Computed Tomography Sinograms for High-Resolution Stroke Imaging

## Objectives
1. Generate a set 2D TM Microwave (MW) Scattered field data over anatomical head models --> Finite, low resolution MW data 
2. Generate a set of sinograms (high frequency, low diffraction projections using parallel beam method) for the same anatomical head model --> Dense, high resolution data
3. Learn the mapping between these two data space: a cross modality mapping to enhance the resolution of MW data.
4. End goal: high resolution imaging of human head and quantitative diagnosis of strokes.

The model, termed as MSSNet, improves the resolution of standard MW imaging (MWI) algorithms.

### Two types of imaging has been considered:
1. Absolute imaging: The whole brain interior is imaged.
2. Differential imaging: Differential change between two measurement instances (Stroke vs no stroke) is imaged.

## Graphical flowchart
<img width="4140" height="1959" alt="GA_figure" src="https://github.com/user-attachments/assets/5458862d-fbef-4725-bcbf-22883465f2e4" />

# Required software
1. Matlab (Image processing toolbox and RF toolbox installed)
2. python, with standard libraries and pytorch

# Steps
## Step 1: Generating the MW and CT dataset


