#!/usr/bin/env python3
"""
Test Cases for Non-English CSV Parsing

This module tests the IntegratedSearchFileCreator's ability to handle:
- Various character encodings (UTF-8, UTF-8-BOM, CP1252, Latin-1, etc.)
- International characters (Chinese, Japanese, Arabic, Hindi, Cyrillic, etc.)
- Mixed encoding scenarios
- Edge cases with special characters and symbols
"""

import unittest
import pandas as pd
import tempfile
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
import io
import sys
import os

# Add the parent directory to Python path to import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import the class we're testing
    from create_integrated_search_file import IntegratedSearchFileCreator
except ImportError as e:
    print(f"Warning: Could not import IntegratedSearchFileCreator: {e}")
    print("Running in standalone mode with manual tests only")
    IntegratedSearchFileCreator = None


class TestNonEnglishCSVParsing(unittest.TestCase):
    """Test suite for non-English CSV parsing capabilities."""
    
    def setUp(self):
        """Set up test workspace."""
        if IntegratedSearchFileCreator is None:
            self.skipTest("IntegratedSearchFileCreator not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir)
        
        # Create required directory structure
        (self.workspace / "input_data" / "data_flat_file").mkdir(parents=True)
        (self.workspace / "input_data" / "table_relations").mkdir(parents=True)
        (self.workspace / "processed_data_store" / "company_mapped_store").mkdir(parents=True)
        
        # Create sample relations.json file
        relations = [
            {
                "from_table": "CompanyMaster",
                "from_column": "Country_Fk_id",
                "to_table": "CountryMaster",
                "to_column": "Id"
            },
            {
                "from_table": "CompanyMaster",
                "from_column": "CompanyScale_Fk_Id",
                "to_table": "ScaleMaster",
                "to_column": "Id"
            }
        ]
        
        relations_path = self.workspace / "input_data" / "table_relations" / "relations.json"
        with open(relations_path, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False)
    
    def tearDown(self):
        """Clean up test workspace."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
    
    def create_csv_with_encoding(self, filepath: Path, data: Dict, encoding: str, 
                                include_bom: bool = False):
        """Helper to create CSV files with specific encodings."""
        df = pd.DataFrame(data)
        
        if include_bom and encoding.lower() in ['utf-8', 'utf8']:
            encoding = 'utf-8-sig'
        
        try:
            df.to_csv(filepath, index=False, encoding=encoding)
        except UnicodeEncodeError:
            # Fallback for encodings that can't handle certain characters
            # Replace problematic characters with closest ASCII equivalent
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.encode(encoding, errors='ignore').str.decode(encoding)
            df.to_csv(filepath, index=False, encoding=encoding)
    
    def test_utf8_chinese_characters(self):
        """Test parsing CSV with Chinese characters in UTF-8."""
        chinese_data = {
            'Id': [1, 2, 3],
            'CompanyName': ['北京科技有限公司', '上海制造企业', '深圳创新公司'],
            'Address': ['北京市朝阳区', '上海市浦东新区', '深圳市南山区'],
            'CityName': ['北京', '上海', '深圳'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20', '2021-03-10'],
            'Country_Fk_id': [1, 1, 1]
        }
        
        # Create CompanyMaster CSV
        company_csv = self.workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, chinese_data, 'utf-8')
        
        # Create CountryMaster CSV
        country_data = {
            'Id': [1],
            'CountryName': ['中国']
        }
        country_csv = self.workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        # Test parsing
        creator = IntegratedSearchFileCreator(str(self.workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should successfully parse Chinese UTF-8 CSV"
        assert 'CompanyMaster' in creator.tables
        assert len(creator.tables['CompanyMaster']) == 3
        
        # Verify Chinese characters are preserved
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert '北京科技有限公司' in company_names
        assert '上海制造企业' in company_names
    
    def test_utf8_bom_japanese_characters(self):
        """Test parsing CSV with Japanese characters and UTF-8 BOM."""
        japanese_data = {
            'Id': [1, 2],
            'CompanyName': ['東京テクノロジー株式会社', '大阪製造業有限会社'],
            'Address': ['東京都渋谷区', '大阪府大阪市'],
            'CityName': ['東京', '大阪'],
            'CompanyRegistrationDate': ['2020-04-01', '2018-07-15'],
            'Country_Fk_id': [2, 2]
        }
        
        company_csv = self.workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, japanese_data, 'utf-8', include_bom=True)
        
        country_data = {
            'Id': [2],
            'CountryName': ['日本']
        }
        country_csv = self.workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(self.workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should successfully parse Japanese UTF-8-BOM CSV"
        
        # Verify Japanese characters are preserved
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert '東京テクノロジー株式会社' in company_names
    
    def test_cp1252_european_characters(self, temp_workspace, sample_relations):
        """Test parsing CSV with European characters in CP1252 encoding."""
        european_data = {
            'Id': [1, 2, 3],
            'CompanyName': ['Société Française', 'Deutsche Größe GmbH', 'Español Ñoño S.L.'],
            'Address': ['París, França', 'München, Deutschland', 'Madrid, España'],
            'CityName': ['París', 'München', 'Madrid'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20', '2021-03-10'],
            'Country_Fk_id': [3, 4, 5]
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, european_data, 'cp1252')
        
        country_data = {
            'Id': [3, 4, 5],
            'CountryName': ['France', 'Deutschland', 'España']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'cp1252')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should successfully parse CP1252 European CSV"
        
        # Verify special characters are preserved
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert 'Société Française' in company_names
        assert 'Deutsche Größe GmbH' in company_names
        assert 'Español Ñoño S.L.' in company_names
    
    def test_latin1_mixed_characters(self, temp_workspace, sample_relations):
        """Test parsing CSV with mixed Latin-1 characters."""
        latin_data = {
            'Id': [1, 2],
            'CompanyName': ['Café & Résumé Ltd', 'Naïve Piñata Co'],
            'Address': ['Zürich, Schweiz', 'São Paulo, Brasil'],
            'CityName': ['Zürich', 'São Paulo'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20'],
            'Country_Fk_id': [6, 7]
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, latin_data, 'latin-1')
        
        country_data = {
            'Id': [6, 7],
            'CountryName': ['Schweiz', 'Brasil']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'latin-1')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should successfully parse Latin-1 CSV"
        
        # Verify accented characters are preserved
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert 'Café & Résumé Ltd' in company_names
        assert 'Naïve Piñata Co' in company_names
    
    def test_cyrillic_characters_utf8(self, temp_workspace, sample_relations):
        """Test parsing CSV with Cyrillic (Russian) characters."""
        cyrillic_data = {
            'Id': [1, 2],
            'CompanyName': ['ООО Технологии', 'АО Производство'],
            'Address': ['Москва, Россия', 'Санкт-Петербург, Россия'],
            'CityName': ['Москва', 'Санкт-Петербург'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20'],
            'Country_Fk_id': [8, 8]
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, cyrillic_data, 'utf-8')
        
        country_data = {
            'Id': [8],
            'CountryName': ['Россия']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should successfully parse Cyrillic UTF-8 CSV"
        
        # Verify Cyrillic characters are preserved
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert 'ООО Технологии' in company_names
        assert 'АО Производство' in company_names
    
    def test_arabic_characters_utf8(self, temp_workspace, sample_relations):
        """Test parsing CSV with Arabic characters."""
        arabic_data = {
            'Id': [1, 2],
            'CompanyName': ['شركة التكنولوجيا', 'مؤسسة الإنتاج'],
            'Address': ['الرياض، السعودية', 'دبي، الإمارات'],
            'CityName': ['الرياض', 'دبي'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20'],
            'Country_Fk_id': [9, 10]
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, arabic_data, 'utf-8')
        
        country_data = {
            'Id': [9, 10],
            'CountryName': ['السعودية', 'الإمارات']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should successfully parse Arabic UTF-8 CSV"
        
        # Verify Arabic characters are preserved
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert 'شركة التكنولوجيا' in company_names
        assert 'مؤسسة الإنتاج' in company_names
    
    def test_hindi_devanagari_utf8(self, temp_workspace, sample_relations):
        """Test parsing CSV with Hindi/Devanagari characters."""
        hindi_data = {
            'Id': [1, 2],
            'CompanyName': ['प्रौद्योगिकी कंपनी', 'निर्माण उद्यम'],
            'Address': ['नई दिल्ली, भारत', 'मुंबई, भारत'],
            'CityName': ['नई दिल्ली', 'मुंबई'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20'],
            'Country_Fk_id': [11, 11]
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, hindi_data, 'utf-8')
        
        country_data = {
            'Id': [11],
            'CountryName': ['भारत']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should successfully parse Hindi UTF-8 CSV"
        
        # Verify Hindi characters are preserved
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert 'प्रौद्योगिकी कंपनी' in company_names
        assert 'निर्माण उद्यम' in company_names
    
    def test_mixed_encoding_files(self, temp_workspace, sample_relations):
        """Test handling multiple CSV files with different encodings."""
        # CompanyMaster in UTF-8 with Chinese
        chinese_data = {
            'Id': [1, 2],
            'CompanyName': ['北京科技公司', '上海制造'],
            'Address': ['北京市', '上海市'],
            'CityName': ['北京', '上海'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20'],
            'Country_Fk_id': [1, 1]
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, chinese_data, 'utf-8')
        
        # CountryMaster in CP1252 with European characters
        country_data = {
            'Id': [1, 2],
            'CountryName': ['中国', 'François']  # This will test fallback for problematic chars
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        try:
            self.create_csv_with_encoding(country_csv, country_data, 'cp1252')
        except UnicodeEncodeError:
            # Expected - create with UTF-8 instead
            self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should handle mixed encoding files gracefully"
        assert len(creator.tables) >= 2
    
    def test_special_characters_and_symbols(self, temp_workspace, sample_relations):
        """Test parsing CSV with special characters, symbols, and emojis."""
        special_data = {
            'Id': [1, 2, 3],
            'CompanyName': ['Tech™ Corp®', 'Data & Analytics©', 'AI Solutions™'],
            'Address': ['123 Main St. #456', 'Suite 789 @ Business Park', 'Floor 10, Tower-A'],
            'CityName': ['San José', 'Montréal', 'São Paulo'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20', '2021-03-10'],
            'Website': ['https://tech™.com', 'www.data&analytics.org', 'ai-solutions.net'],
            'EmailId': ['info@tech™.com', 'contact@data&analytics.org', 'hello@ai-solutions.net'],
            'Country_Fk_id': [1, 2, 3]
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, special_data, 'utf-8')
        
        country_data = {
            'Id': [1, 2, 3],
            'CountryName': ['USA', 'Canada', 'Brazil']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should successfully parse CSV with special characters"
        
        # Verify special characters are preserved
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert 'Tech™ Corp®' in company_names
        assert 'Data & Analytics©' in company_names
    
    def test_corrupted_encoding_fallback(self, temp_workspace, sample_relations):
        """Test fallback mechanism for corrupted or problematic encodings."""
        # Create a file with intentionally mixed/corrupted encoding
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        
        # Manually write file with mixed encoding issues
        with open(company_csv, 'wb') as f:
            # Write header in UTF-8
            f.write('Id,CompanyName,Address,CityName,CompanyRegistrationDate,Country_Fk_id\n'.encode('utf-8'))
            # Write first row in UTF-8
            f.write('1,正常公司,北京市,北京,2020-01-15,1\n'.encode('utf-8'))
            # Write second row with Latin-1 bytes that aren't valid UTF-8
            f.write('2,'.encode('utf-8'))
            f.write('Café résumé'.encode('latin-1'))  # This creates invalid UTF-8 sequence
            f.write(',Paris,Paris,2019-05-20,2\n'.encode('utf-8'))
        
        country_data = {
            'Id': [1, 2],
            'CountryName': ['中国', 'France']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        # Should succeed with fallback encoding
        assert success, "Should handle corrupted encoding with fallback mechanism"
        assert 'CompanyMaster' in creator.tables
    
    def test_empty_and_null_non_english_fields(self, temp_workspace, sample_relations):
        """Test handling of empty and null values in non-English text fields."""
        mixed_data = {
            'Id': [1, 2, 3, 4],
            'CompanyName': ['正常公司', '', '空值公司', None],
            'Address': ['北京市朝阳区', None, '', '上海市'],
            'CityName': ['北京', '', None, '上海'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20', '', '2021-03-10'],
            'Country_Fk_id': [1, 1, None, 1]
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        
        # Create DataFrame and handle None values properly
        df = pd.DataFrame(mixed_data)
        df.to_csv(company_csv, index=False, encoding='utf-8')
        
        country_data = {
            'Id': [1],
            'CountryName': ['中国']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should handle empty/null values in non-English fields"
        
        # Verify the file was created and contains expected data
        output_file = temp_workspace / "processed_data_store" / "company_mapped_store" / "integrated_company_search.json"
        assert output_file.exists()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        companies = data['companies']
        assert len(companies) == 4
        
        # Check that non-English characters are preserved where present
        company_names = [c['company_name'] for c in companies]
        assert '正常公司' in company_names
        assert '空值公司' in company_names
    
    def test_large_non_english_dataset(self, temp_workspace, sample_relations):
        """Test performance with larger dataset containing non-English characters."""
        # Create larger dataset with mixed international content
        large_data = {
            'Id': list(range(1, 101)),
            'CompanyName': [
                f'公司{i}' if i % 4 == 0 else  # Chinese
                f'Société{i}' if i % 4 == 1 else  # French
                f'Компания{i}' if i % 4 == 2 else  # Russian
                f'Company{i}'  # English
                for i in range(1, 101)
            ],
            'Address': [f'Address {i}' for i in range(1, 101)],
            'CityName': [f'City {i}' for i in range(1, 101)],
            'CompanyRegistrationDate': ['2020-01-15'] * 100,
            'Country_Fk_id': [1] * 100
        }
        
        company_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        self.create_csv_with_encoding(company_csv, large_data, 'utf-8')
        
        country_data = {
            'Id': [1],
            'CountryName': ['International']
        }
        country_csv = temp_workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        self.create_csv_with_encoding(country_csv, country_data, 'utf-8')
        
        creator = IntegratedSearchFileCreator(str(temp_workspace))
        success = creator.create_integrated_file()
        
        assert success, "Should handle large dataset with non-English characters"
        assert len(creator.tables['CompanyMaster']) == 100
        
        # Verify sampling of non-English characters
        company_names = creator.tables['CompanyMaster']['CompanyName'].tolist()
        assert any('公司' in name for name in company_names)  # Chinese
        assert any('Société' in name for name in company_names)  # French
        assert any('Компания' in name for name in company_names)  # Russian


def run_manual_test():
    """Manual test runner for development/debugging."""
    import tempfile
    import shutil
    
    print("🚀 Running manual tests for non-English CSV parsing...")
    
    # Create temporary workspace
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    print(f"📁 Using temporary workspace: {workspace}")
    
    try:
        # Setup directory structure
        (workspace / "input_data" / "data_flat_file").mkdir(parents=True)
        (workspace / "input_data" / "table_relations").mkdir(parents=True)
        
        # Create relations.json
        relations = [
            {
                "from_table": "CompanyMaster",
                "from_column": "Country_Fk_id",
                "to_table": "CountryMaster",
                "to_column": "Id"
            }
        ]
        
        relations_path = workspace / "input_data" / "table_relations" / "relations.json"
        with open(relations_path, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False)
        
        # Test 1: Chinese characters
        print("\n🇨🇳 Testing Chinese characters...")
        chinese_data = {
            'Id': [1, 2],
            'CompanyName': ['北京科技有限公司', '上海制造企业'],
            'Address': ['北京市朝阳区', '上海市浦东新区'],
            'CityName': ['北京', '上海'],
            'CompanyRegistrationDate': ['2020-01-15', '2019-05-20'],
            'Country_Fk_id': [1, 1]
        }
        
        company_csv = workspace / "input_data" / "data_flat_file" / "dbo.CompanyMaster.csv"
        pd.DataFrame(chinese_data).to_csv(company_csv, index=False, encoding='utf-8')
        
        country_data = {
            'Id': [1],
            'CountryName': ['中国']
        }
        country_csv = workspace / "input_data" / "data_flat_file" / "dbo.CountryMaster.csv"
        pd.DataFrame(country_data).to_csv(country_csv, index=False, encoding='utf-8')
        
        # Test the parser
        creator = IntegratedSearchFileCreator(str(workspace))
        success = creator.create_integrated_file()
        
        if success:
            print("✅ Chinese characters test passed!")
            
            # Check output
            output_file = workspace / "processed_data_store" / "company_mapped_store" / "integrated_company_search.json"
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"📊 Created file with {len(data['companies'])} companies")
                sample_company = data['companies'][0]
                print(f"📝 Sample company: {sample_company['company_name']}")
                print(f"🏢 Company country: {sample_company.get('country', 'N/A')}")
        else:
            print("❌ Chinese characters test failed!")
        
        print("\n🎉 Manual test completed!")
        
    except Exception as e:
        print(f"❌ Manual test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary workspace")


if __name__ == "__main__":
    # Run manual test if script is executed directly
    run_manual_test()
