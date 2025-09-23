#!/usr/bin/env python3
"""
Create Integrated Company Search File from Flat Files

This script creates a flattened, fast-searchable JSON file with all company data
by reading directly from CSV flat files and using table relations for foreign key resolution.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedSearchFileCreator:
    """Creates integrated search file from flat CSV files using table relations."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.input_data_path = self.base_path / "input_data" / "data_flat_file"
        self.relations_path = self.base_path / "input_data" / "table_relations" / "relations.json"
        self.processed_data_path = self.base_path / "processed_data_store" / "company_mapped_store"
        self.output_path = self.processed_data_path / "integrated_company_search.json"
        
        # Ensure output directory exists
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Load table relations and CSV data
        self.relations = self._load_table_relations()
        self.tables = self._load_csv_tables()
        
    def _load_table_relations(self) -> List[Dict]:
        """Load table relations from JSON file."""
        try:
            logger.info(f"Loading table relations from: {self.relations_path}")
            with open(self.relations_path, 'r', encoding='utf-8') as f:
                relations = json.load(f)
            logger.info(f"✅ Loaded {len(relations)} table relations")
            return relations
        except Exception as e:
            logger.error(f"Failed to load table relations: {e}")
            return []
    
    def _load_csv_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV tables into memory."""
        logger.info("Loading CSV tables...")
        tables = {}
        
        csv_files = list(self.input_data_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Extract table name from filename (remove dbo. prefix and .csv suffix)
                table_name = csv_file.stem.replace('dbo.', '')
                
                # Load CSV with error handling
                df = pd.read_csv(csv_file, encoding='utf-8')
                tables[table_name] = df
                
                logger.info(f"  ✅ {table_name}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to load {csv_file}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(tables)} tables successfully")
        return tables
    
    def _resolve_foreign_key(self, from_table: str, from_column: str, from_value: Any, to_table: str, to_column: str, target_column: str = None) -> str:
        """Resolve foreign key to actual value using table relations."""
        if pd.isna(from_value) or from_value == '' or from_value is None:
            return ""
        
        try:
            # Get the target table
            if to_table not in self.tables:
                return str(from_value)
            
            target_df = self.tables[to_table]
            
            # Find the matching row
            matching_rows = target_df[target_df[to_column] == from_value]
            
            if matching_rows.empty:
                return str(from_value)
            
            # If target_column is specified, return that column's value
            if target_column and target_column in matching_rows.columns:
                return str(matching_rows.iloc[0][target_column])
            
            # Otherwise, try to find a name column
            name_columns = [col for col in matching_rows.columns if 'name' in col.lower() or 'type' in col.lower()]
            if name_columns:
                return str(matching_rows.iloc[0][name_columns[0]])
            
            # Fallback to the original value
            return str(from_value)
            
        except Exception as e:
            logger.debug(f"FK resolution failed for {from_table}.{from_column} -> {to_table}.{to_column}: {e}")
            return str(from_value) if from_value else ""
    
    def _get_foreign_key_mapping(self, table_name: str) -> Dict[str, Dict]:
        """Get foreign key mappings for a table."""
        fk_mappings = {}
        
        for relation in self.relations:
            if relation['from_table'] == table_name:
                fk_mappings[relation['from_column']] = {
                    'to_table': relation['to_table'],
                    'to_column': relation['to_column']
                }
        
        return fk_mappings
    
    def _create_company_record(self, company_row: pd.Series) -> Dict[str, Any]:
        """Create a comprehensive company record from flat file data."""
        try:
            company_id = company_row['Id']
            
            # Initialize company record with basic info
            company_record = {
                'company_ref_no': f"CMP{str(company_id).zfill(3)}",
                'company_id': int(company_id),
                'company_name': str(company_row.get('CompanyName', '')),
                'cin_number': str(company_row.get('CINNumber', '')),
                'pan': str(company_row.get('Pan', '')),
                'registration_date': str(company_row.get('CompanyRegistrationDate', '')),
                'company_status': str(company_row.get('CompanyStatus', '')),
                'company_class': str(company_row.get('CompanyClass', '')),
                'listing_status': str(company_row.get('ListingStatus', '')),
                'company_category': str(company_row.get('CompanyCategory', '')),
                'company_subcategory': str(company_row.get('CompanySubCategory', '')),
                'industrial_classification': str(company_row.get('CompanyIndustrialClassification', '')),
                'other_expertise': str(company_row.get('OtherCompanyCoreExpertise', '')),
                'other_industry_domain': str(company_row.get('OtherCompIndDomain', '')),
                'other_industry_subdomain': str(company_row.get('OtherCompIndSubDomain', '')),
                
                # Contact Information
                'address': str(company_row.get('Address', '')),
                'city': str(company_row.get('CityName', '')),
                'district': str(company_row.get('District', '')),
                'state': str(company_row.get('State', '')),
                'pincode': str(company_row.get('PinCode', '')),
                'email': str(company_row.get('EmailId', '')),
                'poc_email': str(company_row.get('POC_Email', '')),
                'phone': str(company_row.get('Phone', '')),
                'website': str(company_row.get('Website', '')),
            }
            
            # Resolve foreign keys for CompanyMaster
            fk_mappings = self._get_foreign_key_mapping('CompanyMaster')
            
            # Resolve Country
            if 'Country_Fk_id' in company_row and not pd.isna(company_row['Country_Fk_id']):
                country_name = self._resolve_foreign_key('CompanyMaster', 'Country_Fk_id', 
                                                       company_row['Country_Fk_id'], 'CountryMaster', 'Id', 'CountryName')
                company_record['country'] = country_name
            else:
                company_record['country'] = ''
            
            # Resolve Company Scale
            if 'CompanyScale_Fk_Id' in company_row and not pd.isna(company_row['CompanyScale_Fk_Id']):
                scale = self._resolve_foreign_key('CompanyMaster', 'CompanyScale_Fk_Id', 
                                                company_row['CompanyScale_Fk_Id'], 'ScaleMaster', 'Id', 'CompanyScale')
                company_record['company_scale'] = scale
            else:
                company_record['company_scale'] = ''
            
            # Resolve Organization Type
            if 'CompanyType_Fk_Id' in company_row and not pd.isna(company_row['CompanyType_Fk_Id']):
                org_type = self._resolve_foreign_key('CompanyMaster', 'CompanyType_Fk_Id', 
                                                   company_row['CompanyType_Fk_Id'], 'OrganisationTypeMaster', 'Id', 'Organization_Type')
                company_record['organization_type'] = org_type
            else:
                company_record['organization_type'] = ''
            
            # Resolve Core Expertise
            if 'CompanyCoreExpertise_Fk_Id' in company_row and not pd.isna(company_row['CompanyCoreExpertise_Fk_Id']):
                expertise = self._resolve_foreign_key('CompanyMaster', 'CompanyCoreExpertise_Fk_Id', 
                                                    company_row['CompanyCoreExpertise_Fk_Id'], 'CompanyCoreExpertiseMaster', 'Id', 'CoreExpertiseName')
                company_record['core_expertise'] = expertise
            else:
                company_record['core_expertise'] = ''
            
            # Resolve Industry Domain
            if 'IndustryDomain_Fk_Id' in company_row and not pd.isna(company_row['IndustryDomain_Fk_Id']):
                industry = self._resolve_foreign_key('CompanyMaster', 'IndustryDomain_Fk_Id', 
                                                   company_row['IndustryDomain_Fk_Id'], 'IndustryDomainMaster', 'Id', 'IndustryDomainType')
                company_record['industry_domain'] = industry
            else:
                company_record['industry_domain'] = ''
            
            # Resolve Industry Subdomain
            if 'IndustrySubDomain_Fk_Id' in company_row and not pd.isna(company_row['IndustrySubDomain_Fk_Id']):
                subdomain = self._resolve_foreign_key('CompanyMaster', 'IndustrySubDomain_Fk_Id', 
                                                    company_row['IndustrySubDomain_Fk_Id'], 'IndustrySubdomainType', 'Id', 'IndustrySubDomainName')
                company_record['industry_subdomain'] = subdomain
            else:
                company_record['industry_subdomain'] = ''
            
            # Add related data from other tables
            self._add_certifications(company_record, company_id)
            self._add_products(company_record, company_id)
            self._add_rd_facilities(company_record, company_id)
            self._add_test_facilities(company_record, company_id)
            self._add_turnover(company_record, company_id)
            
            # Clean up empty values
            cleaned_record = {}
            for key, value in company_record.items():
                if value is not None and str(value).strip() != '' and not pd.isna(value):
                    cleaned_record[key] = str(value).strip()
                else:
                    cleaned_record[key] = ''
            
            return cleaned_record
            
        except Exception as e:
            logger.error(f"Failed to create company record for ID {company_row.get('Id', 'unknown')}: {e}")
            return {
                'company_ref_no': f"CMP{str(company_row.get('Id', '000')).zfill(3)}",
                'company_id': int(company_row.get('Id', 0)),
                'company_name': str(company_row.get('CompanyName', 'Unknown')),
                'error': str(e)
            }
    
    def _add_certifications(self, company_record: Dict, company_id: int):
        """Add certification information to company record."""
        try:
            if 'CompanyCertificationDetail' not in self.tables:
                return
            
            cert_df = self.tables['CompanyCertificationDetail']
            company_certs = cert_df[cert_df['CompanyMaster_Fk_Id'] == company_id]
            
            if not company_certs.empty:
                cert_row = company_certs.iloc[0]  # Take first certification
                
                # Resolve certification type
                if 'CertificateType_Fk_Id' in cert_row and not pd.isna(cert_row['CertificateType_Fk_Id']):
                    cert_type = self._resolve_foreign_key('CompanyCertificationDetail', 'CertificateType_Fk_Id',
                                                        cert_row['CertificateType_Fk_Id'], 'CertificationTypeMaster', 'Id', 'Cert_Type')
                    company_record['certification_type'] = cert_type
                
                company_record.update({
                    'certification_detail': str(cert_row.get('Certification_Type', '')),
                    'certificate_number': str(cert_row.get('Certificate_No', '')),
                    'certificate_start_date': str(cert_row.get('Certificate_StartDate', '')),
                    'certificate_end_date': str(cert_row.get('Certificate_EndDate', '')),
                })
                
        except Exception as e:
            logger.debug(f"Failed to add certifications for company {company_id}: {e}")
    
    def _add_products(self, company_record: Dict, company_id: int):
        """Add product information to company record."""
        try:
            if 'CompanyProducts' not in self.tables:
                return
            
            products_df = self.tables['CompanyProducts']
            company_products = products_df[products_df['CompanyMaster_FK_Id'] == company_id]
            
            if not company_products.empty:
                product_row = company_products.iloc[0]  # Take first product
                
                # Resolve product type
                if 'ProductType_Fk_Id' in product_row and not pd.isna(product_row['ProductType_Fk_Id']):
                    product_type = self._resolve_foreign_key('CompanyProducts', 'ProductType_Fk_Id',
                                                           product_row['ProductType_Fk_Id'], 'ProductTypeMaster', 'Id', 'ProductTypeName')
                    company_record['product_type'] = product_type
                
                # Resolve defence platform
                if 'DefencePlatform_Fk_Id' in product_row and not pd.isna(product_row['DefencePlatform_Fk_Id']):
                    defence_platform = self._resolve_foreign_key('CompanyProducts', 'DefencePlatform_Fk_Id',
                                                               product_row['DefencePlatform_Fk_Id'], 'DefencePlatformMaster', 'Id', 'Name_of_Defence_Platform')
                    company_record['defence_platform'] = defence_platform
                
                # Resolve PTA Type
                if 'PTAType_Fk_Id' in product_row and not pd.isna(product_row['PTAType_Fk_Id']):
                    pta_type = self._resolve_foreign_key('CompanyProducts', 'PTAType_Fk_Id',
                                                       product_row['PTAType_Fk_Id'], 'PlatformTechAreaMaster', 'Id', 'PTAName')
                    company_record['platform_tech_area'] = pta_type
                
                company_record.update({
                    'product_name': str(product_row.get('ProductName', '')),
                    'product_description': str(product_row.get('ProductDesc', '')),
                    'hsn_code': str(product_row.get('HSNCode', '')),
                    'nsn_number': str(product_row.get('NSNNumber', '')),
                    'items_exported': str(product_row.get('ItemExported', '')),
                })
                
        except Exception as e:
            logger.debug(f"Failed to add products for company {company_id}: {e}")
    
    def _add_rd_facilities(self, company_record: Dict, company_id: int):
        """Add R&D facility information to company record."""
        try:
            if 'CompanyRDFacility' not in self.tables:
                return
            
            rd_df = self.tables['CompanyRDFacility']
            company_rd = rd_df[rd_df['CompanyMaster_FK_ID'] == company_id]
            
            if not company_rd.empty:
                rd_row = company_rd.iloc[0]  # Take first R&D facility
                
                # Resolve R&D category
                if 'RDCategory_Fk_ID' in rd_row and not pd.isna(rd_row['RDCategory_Fk_ID']):
                    rd_category = self._resolve_foreign_key('CompanyRDFacility', 'RDCategory_Fk_ID',
                                                          rd_row['RDCategory_Fk_ID'], 'RDCategoryMaster', 'Id', 'RDCategoryName')
                    company_record['rd_category'] = rd_category
                
                # Resolve R&D subcategory
                if 'RDSubCategory_Fk_Id' in rd_row and not pd.isna(rd_row['RDSubCategory_Fk_Id']):
                    rd_subcategory = self._resolve_foreign_key('CompanyRDFacility', 'RDSubCategory_Fk_Id',
                                                             rd_row['RDSubCategory_Fk_Id'], 'RDSubCategoryMaster', 'Id', 'RDSubCategoryName')
                    company_record['rd_subcategory'] = rd_subcategory
                
                company_record.update({
                    'rd_details': str(rd_row.get('RD_Details', '')),
                    'rd_nabl_accredited': str(rd_row.get('IsNabIAccredited', '')),
                })
                
        except Exception as e:
            logger.debug(f"Failed to add R&D facilities for company {company_id}: {e}")
    
    def _add_test_facilities(self, company_record: Dict, company_id: int):
        """Add testing facility information to company record."""
        try:
            if 'CompanyTestFacility' not in self.tables:
                return
            
            test_df = self.tables['CompanyTestFacility']
            company_test = test_df[test_df['CompanyMaster_FK_ID'] == company_id]
            
            if not company_test.empty:
                test_row = company_test.iloc[0]  # Take first test facility
                
                # Resolve test facility category
                if 'TestFacilityCategory_Fk_Id' in test_row and not pd.isna(test_row['TestFacilityCategory_Fk_Id']):
                    test_category = self._resolve_foreign_key('CompanyTestFacility', 'TestFacilityCategory_Fk_Id',
                                                            test_row['TestFacilityCategory_Fk_Id'], 'TestFacilityCategoryMaster', 'Id', 'CategoryName')
                    company_record['test_category'] = test_category
                
                # Resolve test facility subcategory
                if 'TestFacilitySubCategory_Fk_id' in test_row and not pd.isna(test_row['TestFacilitySubCategory_Fk_id']):
                    test_subcategory = self._resolve_foreign_key('CompanyTestFacility', 'TestFacilitySubCategory_Fk_id',
                                                               test_row['TestFacilitySubCategory_Fk_id'], 'TestFacilitySubCategoryMaster', 'Id', 'SubCategoryName')
                    company_record['test_subcategory'] = test_subcategory
                
                company_record.update({
                    'test_details': str(test_row.get('TestDetails', '')),
                    'test_nabl_accredited': str(test_row.get('IsNabIAccredited', '')),
                })
                
        except Exception as e:
            logger.debug(f"Failed to add test facilities for company {company_id}: {e}")
    
    def _add_turnover(self, company_record: Dict, company_id: int):
        """Add turnover information to company record."""
        try:
            if 'CompanyTurnOver' not in self.tables:
                return
            
            turnover_df = self.tables['CompanyTurnOver']
            company_turnover = turnover_df[turnover_df['Company_FK_Id'] == company_id]
            
            if not company_turnover.empty:
                turnover_row = company_turnover.iloc[0]  # Take first turnover record
                
                company_record.update({
                    'turnover_amount': str(turnover_row.get('Amount', '')),
                    'turnover_year': str(turnover_row.get('YearId', '')),
                })
                
        except Exception as e:
            logger.debug(f"Failed to add turnover for company {company_id}: {e}")
    
    def create_integrated_file(self) -> bool:
        """Create the integrated search file from flat CSV files."""
        try:
            logger.info("🚀 Starting integrated search file creation from flat files...")
            
            # Check if CompanyMaster table exists
            if 'CompanyMaster' not in self.tables:
                logger.error("CompanyMaster table not found!")
                return False
            
            company_df = self.tables['CompanyMaster']
            total_companies = len(company_df)
            logger.info(f"Processing {total_companies} companies from CompanyMaster...")
            
            integrated_data = []
            processed_count = 0
            error_count = 0
            
            # Process each company
            for index, company_row in company_df.iterrows():
                try:
                    # Create comprehensive company record
                    company_record = self._create_company_record(company_row)
                    integrated_data.append(company_record)
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count}/{total_companies} companies...")
                    
                except Exception as e:
                    logger.error(f"Failed to process company at index {index}: {e}")
                    error_count += 1
                    continue
            
            # Create metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'total_companies': len(integrated_data),
                'processed_successfully': processed_count,
                'errors': error_count,
                'source_files': {
                    'flat_files_dir': str(self.input_data_path),
                    'relations_file': str(self.relations_path),
                    'tables_loaded': list(self.tables.keys())
                },
                'schema_version': '2.0',
                'description': 'Integrated company search file created from flat CSV files with resolved foreign keys'
            }
            
            # Create final output structure
            output_data = {
                'metadata': metadata,
                'companies': integrated_data
            }
            
            # Write integrated file
            logger.info(f"Writing integrated file to: {self.output_path}")
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Calculate file size
            file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
            
            logger.info("✅ Integrated search file created successfully!")
            logger.info(f"   📁 File: {self.output_path}")
            logger.info(f"   📊 Size: {file_size_mb:.2f} MB")
            logger.info(f"   🏢 Companies: {len(integrated_data)}")
            logger.info(f"   ✅ Processed: {processed_count}")
            logger.info(f"   ❌ Errors: {error_count}")
            logger.info(f"   📋 Tables Used: {len(self.tables)}")
            
            # Log sample data for verification
            if integrated_data:
                sample_company = integrated_data[0]
                logger.info(f"   📝 Sample Company: {sample_company.get('company_name', 'Unknown')} ({sample_company.get('company_ref_no', 'Unknown')})")
                
                # Log available fields
                non_empty_fields = [k for k, v in sample_company.items() if v and str(v).strip()]
                logger.info(f"   🔍 Sample Fields ({len(non_empty_fields)}): {', '.join(non_empty_fields[:10])}{'...' if len(non_empty_fields) > 10 else ''}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create integrated search file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

def main():
    """Main function to create integrated search file from flat files."""
    creator = IntegratedSearchFileCreator()
    success = creator.create_integrated_file()
    
    if success:
        print("🎉 Integrated search file created successfully from flat files!")
        print("   The file now contains data directly from CSV files with resolved foreign keys!")
    else:
        print("❌ Failed to create integrated search file")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
