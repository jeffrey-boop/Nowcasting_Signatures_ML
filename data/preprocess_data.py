import pandas as pd
from functools import reduce
from pathlib import Path
import numpy as np

# Define a function to upsample quarterly data to monthly and fill with NaNs
def upsample_quarterly_to_monthly(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    df = df.resample('MS').asfreq()
    df.reset_index(inplace=True)  # Reset the index so 'Date' becomes a column again
    return df

# The path to your .xls file
current_dir = Path(__file__).parent
xls_path = current_dir / 'Nowcasting_US_GDP.xls'

sheet_names = ['Monthly', 'Quarterly', 'Monthly,_End_of_Month', 'Monthly,_End_of_Period']  # Add your sheet names here

# Use list comprehension to create a list of DataFrames, upsampling if necessary
dataframes = [
    upsample_quarterly_to_monthly(pd.read_excel(xls_path, sheet_name=sheet_name), 'DATE') if 'Quarterly' in sheet_name
    else pd.read_excel(xls_path, sheet_name=sheet_name)
    for sheet_name in sheet_names
]

# Use reduce to merge all DataFrames on the 'Date' column
all_data_merged = reduce(lambda left, right: pd.merge(left, right, on='DATE', how='outer'), dataframes)

# Sort by Date
all_data_merged.sort_values('DATE', inplace=True)

# Format the 'Date' column to display dates in "yyyy-mm-dd" format
all_data_merged['DATE'] = all_data_merged['DATE'].dt.strftime('%Y-%m-%d')

# Modify column names to remove part after underscore
all_data_merged.columns = [col.split('_')[0] for col in all_data_merged.columns]

sorted_cols = sorted([col for col in all_data_merged.columns if col != 'DATE'])
all_data_merged = all_data_merged[['DATE'] + sorted_cols]

# Replace the first non-NaN zero with NaN for each column, excluding 'Date'
for col in [c for c in all_data_merged.columns if c != 'DATE']:
    first_non_nan_zero_idx = all_data_merged[col].loc[(all_data_merged[col].notna()) & (all_data_merged[col] == 0)].index
    if not first_non_nan_zero_idx.empty:
        all_data_merged.at[first_non_nan_zero_idx[0], col] = np.nan

all_data_merged['UMCSENT'] = all_data_merged['UMCSENT'].replace({0: np.nan})

# Export to a new Excel file
save_dir = current_dir / 'merged_data.csv'
all_data_merged.to_csv(save_dir, index=False)

