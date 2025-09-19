## InsightDB
Project for making ML Query for Database

### Start by creating virutal enviornment

Setup your virtual env
- cd venv/Script
-  .\venv\Scripts\Activate.ps1

### Install sentence transformers & llama-cpp-python

- python.exe .\models\get_sentence_transformer_models.py
- .\flat-file-approach\tools\setup_llama_and_models.ps1


**Note**
---
You can create Symlinks for the model if space issue is there
For this u need to be admin in powershell

```
New-Item -ItemType SymbolicLink `
  -Path "mistral-7b-instruct-v0.2.Q5_K_S.gguf" `
  -Target "D:\CBDPIT\models\mistral-7b-instruct-v0.2.Q5_K_S.gguf"
```

```
New-Item -ItemType SymbolicLink `
  -Path "Qwen2.5-14B-Instruct-Q4_K_M.gguf" `
  -Target "D:\CBDPIT\models\Qwen2.5-14B-Instruct-Q4_K_M.gguf"
```

### Run the following commands
cd InsightDB-v2-main
1. python src\load_config.py
2. python -m src.input_data_processing.generate_grouped_csvs_with_data
3. python -m src.company_index.company_search_api  create 
4. python -m src.query_engine.enhanced_query_with_summary
