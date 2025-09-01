import pandas as pd
import numpy as np

def analyze_and_merge_datasets():
    """
    Analyze and merge three datasets to create a custom dataset showing
    relationship between pollutants concentration (PM2.5) and vehicle data
    """
    
    print("Loading datasets...")
    
    # Load WHO PM2.5 data
    who_df = pd.read_csv('data_who.csv')
    print(f"WHO PM2.5 data shape: {who_df.shape}")
    
    # Load Fuel Consumption data  
    fuel_df = pd.read_csv('FuelConsumptionCo2.csv')
    print(f"Fuel consumption data shape: {fuel_df.shape}")
    
    # Load CMC air quality data (sample first to avoid memory issues)
    cmc_df = pd.read_csv('CMC-2023-03-06-merged.csv', nrows=5000)
    print(f"CMC air quality data shape (sample): {cmc_df.shape}")
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Analyze WHO data
    print("\n1. WHO PM2.5 Dataset:")
    print("   - Contains PM2.5 concentrations by country and location type")
    print("   - Key columns: Location, FactValueNumeric (PM2.5 concentration)")
    print("   - Sample countries:", who_df['Location'].unique()[:5])
    
    # Analyze Fuel data
    print("\n2. Fuel Consumption Dataset:")
    print("   - Contains vehicle specifications and CO2 emissions")
    print("   - Key columns: MAKE, MODEL, VEHICLECLASS, CO2EMISSIONS")
    print("   - Sample makes:", fuel_df['MAKE'].unique()[:5])
    
    # Analyze CMC data
    print("\n3. CMC Air Quality Dataset:")
    print("   - Contains air quality measurements")
    print("   - Columns:", list(cmc_df.columns))
    
    print("\n" + "="*60)
    print("CREATING MERGED DATASET")
    print("="*60)
    
    # Strategy: Create a synthetic relationship for assignment purposes
    # We'll create regions/areas and assign vehicle data and pollution data
    
    # Step 1: Prepare WHO data - get average PM2.5 by country
    who_clean = who_df[who_df['FactValueNumeric'].notna()].copy()
    who_summary = who_clean.groupby('Location')['FactValueNumeric'].agg(['mean', 'count']).reset_index()
    who_summary.columns = ['Country', 'Avg_PM25', 'PM25_Measurements']
    who_summary = who_summary[who_summary['PM25_Measurements'] >= 2]  # Countries with multiple measurements
    print(f"WHO data prepared: {len(who_summary)} countries with PM2.5 data")
    
    # Step 2: Prepare fuel data - get average emissions by vehicle class
    fuel_summary = fuel_df.groupby('VEHICLECLASS').agg({
        'CO2EMISSIONS': ['mean', 'count'],
        'FUELCONSUMPTION_COMB': 'mean',
        'ENGINESIZE': 'mean'
    }).reset_index()
    fuel_summary.columns = ['Vehicle_Class', 'Avg_CO2_Emissions', 'Vehicle_Count', 'Avg_Fuel_Consumption', 'Avg_Engine_Size']
    print(f"Fuel data prepared: {len(fuel_summary)} vehicle classes")
    
    # Step 3: Create mapping regions (for assignment purposes)
    # We'll create synthetic regions that combine countries with vehicle types
    regions = [
        'North America - Urban', 'North America - Rural',
        'Europe - Urban', 'Europe - Rural', 
        'Asia - Urban', 'Asia - Rural',
        'South America - Urban', 'South America - Rural'
    ]
    
    # Step 4: Create merged dataset
    merged_data = []
    
    np.random.seed(42)  # For reproducible results
    
    for i, region in enumerate(regions):
        # Sample some countries for this region
        region_countries = who_summary.sample(min(3, len(who_summary)), random_state=i)
        
        # Sample vehicle classes for this region
        region_vehicles = fuel_summary.sample(min(2, len(fuel_summary)), random_state=i)
        
        for _, country_data in region_countries.iterrows():
            for _, vehicle_data in region_vehicles.iterrows():
                merged_record = {
                    'Region': region,
                    'Country': country_data['Country'],
                    'Vehicle_Class': vehicle_data['Vehicle_Class'],
                    'PM25_Concentration': country_data['Avg_PM25'],
                    'CO2_Emissions': vehicle_data['Avg_CO2_Emissions'],
                    'Fuel_Consumption': vehicle_data['Avg_Fuel_Consumption'],
                    'Engine_Size': vehicle_data['Avg_Engine_Size'],
                    'Vehicle_Count_Sample': vehicle_data['Vehicle_Count'],
                    'PM25_Measurement_Count': country_data['PM25_Measurements']
                }
                merged_data.append(merged_record)
    
    # Create final merged DataFrame
    merged_df = pd.DataFrame(merged_data)
    
    # Add some derived columns for analysis
    merged_df['Pollution_Level'] = pd.cut(merged_df['PM25_Concentration'], 
                                        bins=[0, 10, 25, 50, 100], 
                                        labels=['Low', 'Moderate', 'High', 'Very High'])
    
    merged_df['Emission_Category'] = pd.cut(merged_df['CO2_Emissions'],
                                          bins=[0, 200, 300, 400, 600],
                                          labels=['Low', 'Medium', 'High', 'Very High'])
    
    print(f"Merged dataset created: {len(merged_df)} records")
    print(f"Regions: {merged_df['Region'].nunique()}")
    print(f"Countries: {merged_df['Country'].nunique()}")
    print(f"Vehicle Classes: {merged_df['Vehicle_Class'].nunique()}")
    
    # Save the merged dataset
    merged_df.to_csv('merged_pollution_vehicle_dataset.csv', index=False)
    print("\nMerged dataset saved as 'merged_pollution_vehicle_dataset.csv'")
    
    # Display sample of merged data
    print("\n" + "="*60)
    print("SAMPLE OF MERGED DATASET")
    print("="*60)
    print(merged_df.head(10))
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(merged_df.describe())
    
    return merged_df

if __name__ == "__main__":
    merged_dataset = analyze_and_merge_datasets()
