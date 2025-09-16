python src\load_config.py
python -m src.generate_grouped_csvs_with_data

python -m src.company_index.process_company_data
python -m src.company_index.generate_company_cin_files
python -m src.company_index.create_tfidf_search_index
python -m src.company_index.build_dense_index

python -m unittest src.company_index.test_company_search_api.TestCompanySearchAPI.test_04_search_api_basic
python -m unittest src.company_index.test_company_search_api.TestCompanySearchAPI.test_05_search_api_with_filters


