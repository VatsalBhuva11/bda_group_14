import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_merged_dataset():
    """
    Analyze the merged pollution-vehicle dataset and create visualizations
    """
    try:
        # Load the merged dataset
        df = pd.read_csv('comprehensive_pollution_vehicle_dataset.csv')
        
        print("="*60)
        print("MERGED DATASET ANALYSIS")
        print("="*60)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("\n📊 Key Statistics:")
        print(f"  - Countries: {df['Country'].nunique()}")
        print(f"  - Economic Regions: {df['Economic_Region'].nunique()}")
        print(f"  - Vehicle Makes: {df['Vehicle_Make'].nunique()}")
        print(f"  - Vehicle Classes: {df['Vehicle_Class'].nunique()}")
        print(f"  - Years: {sorted(df['Year'].unique())}")
        
        print("\n🌍 PM2.5 Concentration by Region:")
        region_pm25 = df.groupby('Economic_Region')['PM25_Concentration'].agg(['mean', 'std', 'count'])
        print(region_pm25)
        
        print("\n🚗 Average CO2 Emissions by Vehicle Class:")
        vehicle_co2 = df.groupby('Vehicle_Class')['Avg_CO2_Emissions'].agg(['mean', 'std', 'count'])
        print(vehicle_co2)
        
        print("\n🔗 Correlation Analysis:")
        numeric_cols = ['PM25_Concentration', 'Avg_CO2_Emissions', 'Avg_Fuel_Consumption', 
                       'Avg_Engine_Size', 'Pollution_Impact_Score']
        correlation_matrix = df[numeric_cols].corr()
        print(correlation_matrix)
        
        print("\n📈 Sample Records:")
        print(df.head(10))
        
        # Save analysis results
        analysis_results = {
            'dataset_shape': df.shape,
            'countries': df['Country'].nunique(),
            'regions': df['Economic_Region'].nunique(),
            'vehicle_makes': df['Vehicle_Make'].nunique(),
            'years': sorted(df['Year'].unique()),
            'pm25_by_region': region_pm25.to_dict(),
            'co2_by_vehicle': vehicle_co2.to_dict(),
            'correlations': correlation_matrix.to_dict()
        }
        
        print("\n✅ Analysis complete! Dataset is ready.")
        return df, analysis_results
        
    except FileNotFoundError:
        print("❌ Merged dataset not found. Please run comprehensive_merge.py first.")
        return None, None
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None, None

if __name__ == "__main__":
    df, results = analyze_merged_dataset()
