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
    experiments: list = ["historical", "ssp245", "ssp585"]
):
    """
    Search and download downscaled CORDEX South Asia NetCDF data from ESGF.
    
    Parameters:
    - username (str): ESGF Username (can be loaded from ESGF_USER env variable)
    - password (str): ESGF Password (can be loaded from ESGF_PASS env variable)
    - openid_url (str): ESGF OpenID URL (can be loaded from ESGF_OPENID env variable)
    - output_dir (str): Directory to save downloaded NetCDF (.nc) files.
    - variables (str): Daily variables to download.
      - 'tasmax' (Daily Maximum Near-Surface Air Temperature)
      - 'pr' (Precipitation Flux)
      - 'mrso' (Total Soil Moisture Content)
    - experiments (list): List of scenarios to target (historical, ssp245, ssp585, rcp45, rcp85).
    """
    # 1. Resolve credentials
    username = username or os.environ.get("ESGF_USER")
    password = password or os.environ.get("ESGF_PASS")
    openid_url = openid_url or os.environ.get("ESGF_OPENID")

    if not username or not password or not openid_url:
        print("❌ Error: Missing ESGF credentials. Please provide them or set ESGF_USER, ESGF_PASS, ESGF_OPENID in .env.")
        print("See instructions to register at global nodes (e.g. DKRZ: https://esgf-data.dkrz.de).")
        return False

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from pyesgf.search import SearchConnection
        from pyesgf.logon import LogonManager
    except ImportError:
        print("❌ Error: 'esgf-pyclient' library is not installed. Run 'pip install esgf-pyclient' first.")
        return False

    print("🔐 Authenticating with ESGF Identity Provider...")
    try:
        lm = LogonManager()
        # Bootstrap login using OpenID credentials
        lm.logon_with_openid(openid=openid_url, password=password, bootstrap=True)
        if not lm.is_logged_on():
            print("❌ Authentication failed. Double-check your password and OpenID URL.")
            return False
        print("✅ Successfully authenticated with ESGF.")
    except Exception as e:
        print(f"❌ ESGF authentication error: {e}")
        return False

    # Establish search connection (using stable global ESGF Index Node NCI)
    # Since ESGF is federated, querying any index node searches the entire federation (including IITM CCCR).
    print("📡 Connecting to ESGF Search API index node...")
    conn = SearchConnection('https://esgf.nci.org.au/esg-search', distrib=True)

    for var in variables:
        for exp in experiments:
            print(f"\n🔍 Searching for CORDEX South Asia | Variable: {var} | Scenario: {exp} ...")
            
            # Constraints for downscaled CORDEX South Asia domain (WAS-44 or WAS-22)
            ctx = conn.new_context(
                project='CORDEX',
                domain='WAS-44',         # West Asia / South Asia grid (50km resolution)
                experiment=exp,
                variable=var,
                time_frequency='day',
                latest=True
            )
            
            hits = ctx.hit_count
            print(f"   Found {hits} matching datasets.")
            if hits == 0:
                continue

            # Fetch the first matching dataset and search its files
            datasets = ctx.search(limit=1)
            if not datasets:
                continue
            
            dataset = datasets[0]
            print(f"   Targeting Dataset: {dataset.dataset_id}")
            
            files = dataset.file_context().search()
            print(f"   Found {len(files)} files in dataset. Starting downloads...")
            
            # Download files to target directory
            for f in files:
                filename = f.filename
                dest_path = os.path.join(output_dir, filename)
                
                if os.path.exists(dest_path):
                    print(f"   ⏭️ File already exists, skipping: {filename}")
                    continue
                
                download_url = f.download_url
                print(f"   📥 Downloading {filename} ...")
                try:
                    # Retrieve authenticated session from LogonManager to bypass logins
                    session = lm.get_session()
                    response = session.get(download_url, stream=True, timeout=60)
                    if response.status_code == 200:
                        with open(dest_path, "wb") as out_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                out_file.write(chunk)
                        print(f"   ✅ Saved to {dest_path}")
                    else:
                        print(f"   ❌ HTTP {response.status_code} error downloading file.")
                except Exception as e:
                    print(f"   ❌ Error downloading {filename}: {e}")
                    
    return True

if __name__ == "__main__":
    # Test script loading credentials
    download_cordex_data()
