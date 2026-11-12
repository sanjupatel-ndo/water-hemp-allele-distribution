# water-hemp-allele-distribution
Creating a gradient forest model to predict the allele distribution of water hemp 

A primary goal of this repository is to build out the waterhemp sample datasets. This includes herbicide usage, weather data, and land use data.

To properly load the dataset, run the uncertainty prep pipeline on the sample dataset. Then run the climateNA or PRISM pipeline with a new data set file (Data_complete.csv for example) as the export. Finally, run the rest of the pipelines with the new file as both import and export.