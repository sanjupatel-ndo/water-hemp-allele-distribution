"""
Kreiner Lab - September 14th 2026
Resampling pipeline
Primary contributor(s): Sanju Patel

Objective: Pull environmental data for samples with varying degrees of location 
uncertainty

Methodology: For each sample, draw a circle where the sample may have been
collected. For each environmental variable, randomly sample some number of 
observations within this radius. For this first pass, report the mean and 
standard deviation of the sampling distribution for each environmental variable, 
for each sample.

Data requirements:
    Inputs: Waterhemp sample with a latitude and longtitude
"""

