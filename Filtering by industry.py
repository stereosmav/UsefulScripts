import pandas as pd

df = pd.read_excel("All_Data_Raw_v3.xlsx")
df.info

# Check the initial data
print("Initial Data:")
print(df.head())

# Keep only the latest year (Index) for each company
df_latest_year = df.sort_values(by='index', ascending=False).groupby('Company_Name').head(1)

# Check the data with only the latest year for each company
print("\nData with Latest Year for Each Company:")
print(df_latest_year.head())

# Sort the DataFrame by NACE_LVL1 and Total_Net_Sales in descending order
df_sorted = df_latest_year.sort_values(by=['NACE_LVL1', 'Total Net Sales'], ascending=[True, False])

# Group by NACE_LVL1 and get the top 10 entries for each group
top_10_per_industry = df_sorted.groupby('NACE_LVL1').head(10)

# Check the final grouped data
print("\nTop 10 per Industry:")
for industry, group in top_10_per_industry.groupby('NACE_LVL1'):
    print(f"\nIndustry: {industry}")
    print(group)

# Optionally, save the result to a new Excel file
output_file_path = 'top_10_per_industry_latest_year.xlsx'
top_10_per_industry.to_excel(output_file_path, index=False)
print(f"\nTop 10 per industry saved to {output_file_path}")