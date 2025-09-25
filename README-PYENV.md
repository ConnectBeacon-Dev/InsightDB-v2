## InsightDB-v2: Simplified version of search

### Creating the enviornment
- python bootstrap_pyenv_local.py create

### Check if all is good
- python bootstrap_pyenv_local.py doctor

### Install requirements
- python bootstrap_pyenv_local.py pip -- install -r requirements.txt

### Install the application
- python bootstrap_pyenv_local.py run -- .\run_pyenv_pipeline.py


### Run the questions
python bootstrap_pyenv_local.py run -- .\test_pyenv_integrated_query_summary.py --query "location of Company_003"

python bootstrap_pyenv_local.py run -- .\test_integrated_query_summary.py --query "location of Company_003"

### Other questions
python bootstrap_pyenv_local.py run -- .\test_pyen_integrated_query_summary.py --query "list all companies in State5"

python bootstrap_pyenv_local.py run -- .\test_pyen_integrated_query_summary.py --query "Companies having ISO Certificate"

python bootstrap_pyenv_local.py run -- .\test_pyen_integrated_query_summary.py --query "List Companies having R&D facility"

python bootstrap_pyenv_local.py run -- .\test_pyen_integrated_query_summary.py --query "List small scale Companies having R&D facility"

python bootstrap_pyenv_local.py run -- .\test_pyen_integrated_query_summary.py --query "list compnaies making defence equipments"

python bootstrap_pyenv_local.py run -- .\test_pyen_integrated_query_summary.py --query "List Companies having R&D facility"

python bootstrap_pyenv_local.py run -- .\test_pyen_integrated_query_summary.py --query "List Companies having R&D facility and testing facility"

python bootstrap_pyenv_local.py run -- .\test_pyen_integrated_query_summary.py --query "List Companies having R&D facility and testing facility for electrical testing"


### Clean up
- python bootstrap_pyenv_local.py clean

### Additional tests (optional)
- python bootstrap_pyenv_local.py clean
- python bootstrap_pyenv_local.py run -- src\load_config.py
- python bootstrap_pyenv_local.py run -- -m src.company_index.create_integrated_search_file
- python bootstrap_pyenv_local.py run -- -m src.company_index.company_tfidf_api
- python bootstrap_pyenv_local.py run -- -m src.query_engine.enhanced_query_with_summary
- python bootstrap_pyenv_local.py run -- .\test_pyenv_integrated_query_summary.py
- python bootstrap_pyenv_local.py run -- .\test_pyenv_integrated_query_summary.py --query "location of Company_003"
