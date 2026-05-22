import os
from dotenv import load_dotenv

# Load credentials from .env if present
load_dotenv()

def download_cordex_data(
    username: str = None,
    password: str = None,
    openid_url: str = None,
    output_dir: str = "data/raw/netcdf",
    variables: list = ["tasmax", "pr", "mrso"],
    experiments: list = ["historical", "ssp245", "ssp585"],
    max_files_per_dataset: int = None
):
    """
    Search and download downscaled CORDEX South Asia NetCDF data from ESGF.
    
    Parameters:
    - username (str): ESGF Username
    - password (str): ESGF Password
    - openid_url (str): ESGF OpenID URL
    - output_dir (str): Directory to save downloaded NetCDF (.nc) files.
    - variables (list): Daily variables to download.
    - experiments (list): List of scenarios to target (historical, ssp245, ssp585, rcp45, rcp85).
    - max_files_per_dataset (int): Limit files downloaded per dataset (useful for testing).
    """
    import os
    os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "1"
    
    # 1. Resolve credentials
    username = username or os.environ.get("ESGF_USER")
    password = password or os.environ.get("ESGF_PASS")
    openid_url = openid_url or os.environ.get("ESGF_OPENID")

    use_auth = True
    if not username or not password or not openid_url:
        print("⚠️ Warning: Missing or incomplete ESGF credentials. Proceeding in ANONYMOUS download mode.")
        print("We will attempt to download from active public replica nodes (e.g., SMHI Swedish node, DKRZ German node).")
        use_auth = False

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from pyesgf.search import SearchConnection
        from pyesgf.logon import LogonManager
    except ImportError:
        print("❌ Error: 'esgf-pyclient' library is not installed. Run 'pip install esgf-pyclient' first.")
        return False

    lm = None
    if use_auth:
        print("🔐 Authenticating with ESGF Identity Provider...")
        try:
            lm = LogonManager()
            lm.logon_with_openid(openid=openid_url, password=password, bootstrap=True)
            if not lm.is_logged_on():
                print("❌ Authentication failed. Double-check your password and OpenID URL.")
                return False
            print("✅ Successfully authenticated with ESGF.")
        except Exception as e:
            print(f"❌ ESGF authentication error: {e}. Falling back to ANONYMOUS mode.")
            use_auth = False

    # Establish search connection (using stable global ESGF Index Node NCI)
    print("📡 Connecting to ESGF Search API index node...")
    conn = SearchConnection('https://esgf.nci.org.au/esg-search', distrib=True)

    # Map CMIP6 SSP names to CMIP5 RCP names if querying CORDEX
    exp_mapping = {
        "ssp245": "rcp45",
        "ssp585": "rcp85",
        "ssp126": "rcp26"
    }

    for var in variables:
        for exp in experiments:
            search_exp = exp_mapping.get(exp.lower(), exp.lower())
            print(f"\n🔍 Searching for CORDEX South Asia | Variable: {var} | Scenario: {search_exp} (requested: {exp}) ...")
            
            # Constraints for downscaled CORDEX South Asia domain (WAS-44 or WAS-22)
            ctx = conn.new_context(
                project='CORDEX',
                domain='WAS-44',         # West Asia / South Asia grid (50km resolution)
                experiment=search_exp,
                variable=var,
                time_frequency='day',
                latest=True,
                facets='project,domain,experiment,variable,time_frequency'
            )
            
            hits = ctx.hit_count
            print(f"   Found {hits} matching datasets.")
            if hits == 0:
                continue

            # Fetch matching datasets and filter out the broken/404 IITM node
            datasets = ctx.search(limit=20)
            if not datasets:
                continue
            
            target_dataset = None
            for ds in datasets:
                if "esg-cccr.tropmet.res.in" not in ds.dataset_id:
                    target_dataset = ds
                    break
            
            if not target_dataset:
                print("   ⚠️ No replica datasets found on active (non-IITM) nodes.")
                continue
                
            print(f"   Targeting Dataset: {target_dataset.dataset_id}")
            
            try:
                files = target_dataset.file_context().search()
                print(f"   Found {len(files)} files in dataset. Starting downloads...")
            except Exception as e:
                print(f"   ❌ Error fetching files list: {e}")
                continue
            
            # Download files to target directory
            downloaded = 0
            for f in files:
                if max_files_per_dataset is not None and downloaded >= max_files_per_dataset:
                    print(f"   ✋ Reached limit of {max_files_per_dataset} files per dataset. Skipping remainder.")
                    break
                    
                filename = f.filename
                dest_path = os.path.join(output_dir, filename)
                
                if os.path.exists(dest_path):
                    print(f"   ⏭️ File already exists, skipping: {filename}")
                    downloaded += 1
                    continue
                
                download_url = f.download_url
                print(f"   📥 Downloading {filename} ...")
                try:
                    import requests
                    if lm and use_auth:
                        session = lm.get_session()
                    else:
                        session = requests.Session()
                        
                    response = session.get(download_url, stream=True, timeout=60)
                    if response.status_code == 200:
                        with open(dest_path, "wb") as out_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                out_file.write(chunk)
                        print(f"   ✅ Saved to {dest_path}")
                        downloaded += 1
                    else:
                        print(f"   ❌ HTTP {response.status_code} error downloading file.")
                except Exception as e:
                    print(f"   ❌ Error downloading {filename}: {e}")
                    
    return True

if __name__ == "__main__":
    # Test script loading credentials
    download_cordex_data()
