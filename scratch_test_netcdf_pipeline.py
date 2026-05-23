import os
import sys
import shutil

# Make sure we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.pipeline.netcdf_downloader import download_cordex_data
from src.pipeline.netcdf_parser import parse_and_process_netcdf

def main():
    test_raw_dir = "data/test_raw/netcdf"
    test_processed_dir = "data/test_processed"
    
    # Clean up old test data if any
    if os.path.exists(test_raw_dir):
        shutil.rmtree(test_raw_dir)
    if os.path.exists(test_processed_dir):
        shutil.rmtree(test_processed_dir)
        
    os.makedirs(test_raw_dir, exist_ok=True)
    os.makedirs(test_processed_dir, exist_ok=True)
    
    print("🚀 STEP 1: Running download_cordex_data on a single file...")
    # Search for tasmax in ssp245 (rcp45) and download at most 1 file
    download_success = download_cordex_data(
        output_dir=test_raw_dir,
        variables=["tasmax"],
        experiments=["ssp245"],
        max_files_per_dataset=1
    )
    
    if not download_success:
        print("❌ Download failed.")
        return
        
    # Check if files downloaded
    files = os.listdir(test_raw_dir)
    print(f"📁 Downloaded files: {files}")
    if not files:
        print("❌ No files downloaded.")
        return
        
    print("\n🚀 STEP 2: Running parse_and_process_netcdf on downloaded test files...")
    parse_success = parse_and_process_netcdf(
        netcdf_dir=test_raw_dir,
        output_dir=test_processed_dir,
        scenarios=["ssp245", "rcp45"]
    )
    
    if not parse_success:
        print("❌ Parsing failed.")
        return
        
    # Check processed CSV output
    processed_files = os.listdir(test_processed_dir)
    print(f"📁 Processed files count: {len(processed_files)}")
    if processed_files:
        print(f"📄 Sample processed file: {processed_files[0]}")
        import pandas as pd
        sample_path = os.path.join(test_processed_dir, processed_files[0])
        df = pd.read_csv(sample_path)
        print("\n📊 Sample dataframe header:")
        print(df.head())
        print(f"\n✅ Pipeline verification completed successfully! Total processed files: {len(processed_files)}")
    else:
        print("❌ No processed CSV files found.")

if __name__ == "__main__":
    main()
