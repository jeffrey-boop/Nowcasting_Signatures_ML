# How to get updated data from the FRED database

1. Go to https://fredaccount.stlouisfed.org/public/datalist/5564 to access the saved data list of the 34 input variables in this project.

2. Click "Download Data" and select the "Zipped Excel" format.

3. You should get a file named "Nowcasting_US_GDP.xls". Copy that file into the current folder ("data" folder).

4. Run "preprocess_data.py" to preprocess the data.

5. You should get a file named "merged_data.csv" in the parent/root folder (not the "data" folder), which is the preprocessed input data.