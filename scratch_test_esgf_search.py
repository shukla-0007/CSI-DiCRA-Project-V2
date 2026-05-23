import os
from pyesgf.search import SearchConnection

os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "1"

def test_variables_search():
    node = 'https://esgf.nci.org.au/esg-search'
    variables = ["tasmax", "pr", "mrso"]
    exps = ["historical", "rcp45"]
    domain = "WAS-44"
    
    print(f"📡 Connecting to {node}...")
    try:
        conn = SearchConnection(node, distrib=True)
        for var in variables:
            for exp in exps:
                ctx = conn.new_context(
                    project='CORDEX',
                    domain=domain,
                    experiment=exp,
                    variable=var,
                    time_frequency='day',
                    latest=True,
                    facets='project,domain,experiment,variable,time_frequency'
                )
                
                hits = ctx.hit_count
                datasets = ctx.search(limit=20)
                non_iitm = [d for d in datasets if "esg-cccr.tropmet.res.in" not in d.dataset_id]
                
                print(f"🔍 Var: {var} | Exp: {exp} -> Total Hits: {hits} | Non-IITM: {len(non_iitm)}")
                for idx, dataset in enumerate(non_iitm[:1]):
                    print(f"   📄 Non-IITM Dataset: {dataset.dataset_id}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_variables_search()
