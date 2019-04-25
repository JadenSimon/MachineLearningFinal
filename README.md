Final project for CS5350/6350.

Created by: Jaden Simon and Kyle Price

Uses machine learning techniques to predict future air pollution in Salt Lake City.
Weather data from MesoWest is labeled with the EPA's Air Quality Index (AQI) for each day in the dataset.
Since this is time series forecasting, we use the AQI from each day as a feature in order to predict the next day's AQI.


AQI can be between 0 and 500. The EPA also has categorized certain ranges:
[0, 50] - Good
[51, 100] - Moderate
[101, 150] - Unhealth for Sensistive Groups
[151, 200] - Unhealthy
[201, 300] - Very Unhealthy
[301, 500] - Hazardous

The latter two categories will be ommitted due to being outliers. Our current dataset has only a single 'Hazardous' and 5 'Very Unhealthy' data points out of 7612 examples. They will instead be grouped into the 'Unhealthy' category.

Simple run "run.sh" to execute.
