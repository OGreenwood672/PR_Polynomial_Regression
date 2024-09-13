

# How to use

Step 1
Place .mtn files into folder called raw-mtn
For improved results, make sure all the files are good clean examples of single journeys.

Step 2
Run the file mtn-to-csv. This converts all the files to csv files which are more useable for pandas (python library)
The files should now all be in mtn folder

Step 3
Run the analysis file. This will take a couple of minutes. This should populate the results-a and results-b folder. The difference is which scenario the single journey is (A or B). Scenarios C, D and E are ignored.

Step 4
Run the linear-regression-data-collection-acc file, this will split the data in periods and populate the lr_data folder

Step 5
Run the linear-regression-acc file. This should produce example outputs at the bottom of the file. To rerun the model without retraining, run the final cell in the notebook.

Step 6
If you wish to run a different model (phase vs period), delete all the files in lr_data and run the (linear-regression-data-collection-acc or linear-regression-data-collection depending on periods vs phases)
If you want all new data for the model, delete all the files in all the folders.

Email for qns: greenwood672.dev@gmail.com
