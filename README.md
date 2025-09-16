## InsightDB
Project for making ML Query for Database

### Start by creating virutal enviornment

- baseFolder = CurrentWorkinDirectory
- .\setup_env.ps1
- Models
  - Either:
     Copy-Item ..\gguf* -Destination .\flat-file-approach\models -Force
  - Or:
     SymbolicLink (command at bottom)
     
- python.exe .\flat-file-approach\models\get_sentence_transformer_models.py
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