## InsightDB-v2: Simplified version of search

### Creating the enviornment
- python bootstrap_conda_local.py create

### Check if all is good
- python bootstrap_conda_local.py doctor

### Install requirements
- python bootstrap_conda_local.py pip -- install -r requirements.txt

### Install the application
- python bootstrap_conda_local.py run -- python .\run_pipeline.py


### Run the questions
python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "List Companies having R&D facility and testing facility for electrical testing"

python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "location of Company_003"

### Other questions
python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "list all companies in State5"

python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "Companies having ISO Certificate"

python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "List Companies having R&D facility"

python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "List small scale Companies having R&D facility"

python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "list compnaies making defence equipments"

python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "List Companies having R&D facility"

python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "List Companies having R&D facility and testing facility"

python bootstrap_conda_local.py run -- python test_integrated_query_summary.py --query "List Companies having R&D facility and testing facility for electrical testing"


### Clean up
python bootstrap_conda_local.py clean

### Additional tests (optional)
- python bootstrap_conda_local.py clean