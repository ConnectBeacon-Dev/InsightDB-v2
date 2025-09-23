## InsightDB-v2

### Simplified version of search
- Install conda for windows
  - For Windows: https://www.youtube.com/watch?v=i0DCPOiNK4A
  - CLI Steps
  ```
  curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
  start /wait "" .\miniconda.exe /S
  del .\miniconda.exe
  ```

### Additional commands
- Activating conda: ```conda activate llcpp```
- Deactivating conda: ```conda deactivate llcpp```

### Steps for setup
- python .\automate_setup.py

### Steps for preparing the base
- python .\run_pipeline.py

### Asking Questions
```
python test_integrated_query_summary.py --query "location of Company_003"

python test_integrated_query_summary.py --query "list all companies in State5"

python test_integrated_query_summary.py --query "Companies having ISO Certificate"

python test_integrated_query_summary.py --query "List Companies having R&D facility"

python test_integrated_query_summary.py --query "List small scale Companies having R&D facility"

python test_integrated_query_summary.py --query "list compnaies making defence equipments"

python test_integrated_query_summary.py --query "List Companies having R&D facility"

python test_integrated_query_summary.py --query "List Companies having R&D facility and testing facility"

python test_integrated_query_summary.py --query "List Companies having R&D facility and testing facility for electrical testing"

```
