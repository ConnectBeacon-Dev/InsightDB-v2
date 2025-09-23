#!/usr/bin/env python3
"""
Create Integrated Company Search File from Flat Files

This script creates a flattened, fast-searchable JSON file with all company data
by reading directly from CSV flat files and using table relations for foreign key resolution.
- Robust CSV loader with multi-encoding fallback (UTF-8, UTF-8-SIG, CP1252, Latin-1, ISO-8859-1)
- Optional chardet-based detection if installed
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# Optional: use chardet for smarter encoding detection if available
try:
    import chardet  # type: ignore
    _HAS_CHARDET = True
except Exception:
    _HAS_CHARDET = False

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

    # ---------------------------- Robust I/O helpers ----------------------------

    def _detect_encoding(self, file_path: Path, sample_size: int = 200_000) -> str | None:
        """Try to detect encoding using chardet (if available). Returns encoding or None."""
        if not _HAS_CHARDET:
            return None
        try:
            with open(file_path, "rb") as f:
                raw = f.read(sample_size)
            res = chardet.detect(raw)  # {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}
            enc = (res or {}).get("encoding")
            if enc:
                logger.info(f"  ↪ chardet suggests encoding='{enc}' (conf={res.get('confidence')}) for {file_path.name}")
            return enc
        except Exception as e:
            logger.debug(f"chardet detection failed for {file_path}: {e}")
            return None

    def _read_csv_robust(self, csv_file: Path) -> pd.DataFrame:
        """
        Read a CSV file with robust multi-encoding fallback.
        Tries chardet’s guess first (if present), then common encodings.
        """
        # Prefer comma; if your CSVs vary, you can set sep=None and engine='python' to sniff.
        # sep=None can be slower; keeping default comma unless you need sniffing.
        tried: List[str] = []
        last_err: Exception | None = None

        # 1) Try chardet guess first (if any)
        guess = self._detect_encoding(csv_file)
        encodings = [guess] if guess else []
        # 2) Add prioritized encodings
        encodings += ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]

        for enc in encodings:
            if not enc:
                continue
            try:
                df = pd.read_csv(
                    csv_file,
                    encoding=enc,
                    # If your CSVs have tricky delimiters, switch to engine="python" and optionally sep=None to sniff
                    # engine="python",
                    # sep=None,
                    # If you're on pandas >= 2.0, you could add: encoding_errors="replace",
                )
                logger.info(f"  ✅ Loaded {csv_file.name} with encoding='{enc}': {len(df)} rows, {len(df.columns)} cols")
                return df
            except Exception as e:
                tried.append(enc)
                last_err = e
                logger.debug(f"  ↪ failed with encoding='{enc}' for {csv_file.name}: {e}")

        # 3) Final fallback: try python engine + sep=None (sniff) with Latin-1 (very permissive)
        try:
            df = pd.read_csv(csv_file, engine="python", sep=None, encoding="latin1")
            logger.info(f"  ✅ Loaded {csv_file.name} via fallback sniff/latin1: {len(df)} rows, {len(df.columns)} cols")
            return df
        except Exception as e:
            tried.append("python+sniff+latin1")
            last_err = e

        raise RuntimeError(
            f"Failed to read {csv_file} with encodings tried: {', '.join(encodings)}; "
            f"also tried python+sniff+latin1. Last error: {last_err}"
        )

    # ---------------------------- Load relations/tables ----------------------------

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
        """Load all CSV tables into memory (robust to non-English characters)."""
        logger.info("Loading CSV tables...")
        tables: Dict[str, pd.DataFrame] = {}

        csv_files = list(self.input_data_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            try:
                # Extract table name from filename (remove dbo. prefix and .csv suffix)
                table_name = csv_file.stem.replace('dbo.', '')
                df = self._read_csv_robust(csv_file)
                tables[table_name] = df
            except Exception as e:
                logger.error(f"  ❌ Failed to load {csv_file.name}: {e}")
                continue

        logger.info(f"✅ Loaded {len(tables)} tables successfully")
        return tables

    # ---------------------------- FK resolution helpers ----------------------------

    def _resolve_foreign_key(
        self,
        from_table: str,
        from_column: str,
        from_value: Any,
        to_table: str,
        to_column: str,
        target_column: str | None = None,
    ) -> str:
        """Resolve foreign key to actual value using table relations."""
        if pd.isna(from_value) or from_value == '' or from_value is None:
            return ""

        try:
            if to_table not in self.tables:
                return str(from_value)

            target_df = self.tables[to_table]
            matching_rows = target_df[target_df[to_column] == from_value]

            if matching_rows.empty:
                return str(from_value)

            if target_column and target_column in matching_rows.columns:
                return str(matching_rows.iloc[0][target_column])

            name_columns = [col for col in matching_rows.columns if 'name' in col.lower() or 'type' in col.lower()]
            if name_columns:
                return str(matching_rows.iloc[0][name_columns[0]])

            return str(from_value)

        except Exception as e:
            logger.debug(f"FK resolution failed for {from_table}.{from_column} -> {to_table}.{to_column}: {e}")
            return str(from_value) if from_value else ""

    def _get_foreign_key_mapping(self, table_name: str) -> Dict[str, Dict]:
        """Get foreign key mappings for a table."""
        fk_mappings: Dict[str, Dict] = {}
        for relation in self.relations:
            if relation['from_table'] == table_name:
                fk_mappings[relation['from_column']] = {
                    'to_table': relation['to_table'],
                    'to_column': relation['to_column']
                }
        return fk_mappings

    # ---------------------------- Record assembly ----------------------------

    def _create_company_record(self, company_row: pd.Series) -> Dict[str, Any]:
        """Create a comprehensive company record from flat file data."""
        try:
            company_id = company_row['Id']

            # Initialize company record with basic info
            company_record: Dict[str, Any] = {
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
            _ = self._get_foreign_key_mapping('CompanyMaster')  # retained for future logic

            # Resolve Country
            if 'Country_Fk_id' in company_row and not pd.isna(company_row['Country_Fk_id']):
                country_name = self._resolve_foreign_key(
                    'CompanyMaster', 'Country_Fk_id',
                    company_row['Country_Fk_id'], 'CountryMaster', 'Id', 'CountryName'
                )
                company_record['country'] = country_name
            else:
                company_record['country'] = ''

            # Resolve Company Scale
            if 'CompanyScale_Fk_Id' in company_row and not pd.isna(company_row['CompanyScale_Fk_Id']):
                scale = self._resolve_foreign_key(
                    'CompanyMaster', 'CompanyScale_Fk_Id',
                    company_row['CompanyScale_Fk_Id'], 'ScaleMaster', 'Id', 'CompanyScale'
                )
                company_record['company_scale'] = scale
            else:
                company_record['company_scale'] = ''

            # Resolve Organization Type
            if 'CompanyType_Fk_Id' in company_row and not pd.isna(company_row['CompanyType_Fk_Id']):
                org_type = self._resolve_foreign_key(
                    'CompanyMaster', 'CompanyType_Fk_Id',
                    company_row['CompanyType_Fk_Id'], 'OrganisationTypeMaster', 'Id', 'Organization_Type'
                )
                company_record['organization_type'] = org_type
            else:
                company_record['organization_type'] = ''

            # Resolve Core Expertise
            if 'CompanyCoreExpertise_Fk_Id' in company_row and not pd.isna(company_row['CompanyCoreExpertise_Fk_Id']):
                expertise = self._resolve_foreign_key(
                    'CompanyMaster', 'CompanyCoreExpertise_Fk_Id',
                    company_row['CompanyCoreExpertise_Fk_Id'], 'CompanyCoreExpertiseMaster', 'Id', 'CoreExpertiseName'
                )
                company_record['core_expertise'] = expertise
            else:
                company_record['core_expertise'] = ''

            # Resolve Industry Domain
            if 'IndustryDomain_Fk_Id' in company_row and not pd.isna(company_row['IndustryDomain_Fk_Id']):
                industry = self._resolve_foreign_key(
                    'CompanyMaster', 'IndustryDomain_Fk_Id',
                    company_row['IndustryDomain_Fk_Id'], 'IndustryDomainMaster', 'Id', 'IndustryDomainType'
                )
                company_record['industry_domain'] = industry
            else:
                company_record['industry_domain'] = ''

            # Resolve Industry Subdomain
            if 'IndustrySubDomain_Fk_Id' in company_row and not pd.isna(company_row['IndustrySubDomain_Fk_Id']):
                subdomain = self._resolve_foreign_key(
                    'CompanyMaster', 'IndustrySubDomain_Fk_Id',
                    company_row['IndustrySubDomain_Fk_Id'], 'IndustrySubdomainType', 'Id', 'IndustrySubDomainName'
                )
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
            cleaned_record: Dict[str, Any] = {}
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
                'company_id': int(company_row.get('Id', 0)) if str(company_row.get('Id', '0')).isdigit() else 0,
                'company_name': str(company_row.get('CompanyName', 'Unknown')),
                'error': str(e)
            }

    # ---------------------------- Related data adders ----------------------------

    def _add_certifications(self, company_record: Dict, company_id: int):
        try:
            if 'CompanyCertificationDetail' not in self.tables:
                return
            cert_df = self.tables['CompanyCertificationDetail']
            company_certs = cert_df[cert_df['CompanyMaster_Fk_Id'] == company_id]
            if not company_certs.empty:
                cert_row = company_certs.iloc[0]
                if 'CertificateType_Fk_Id' in cert_row and not pd.isna(cert_row['CertificateType_Fk_Id']):
                    cert_type = self._resolve_foreign_key(
                        'CompanyCertificationDetail', 'CertificateType_Fk_Id',
                        cert_row['CertificateType_Fk_Id'], 'CertificationTypeMaster', 'Id', 'Cert_Type'
                    )
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
        try:
            if 'CompanyProducts' not in self.tables:
                return
            products_df = self.tables['CompanyProducts']
            company_products = products_df[products_df['CompanyMaster_FK_Id'] == company_id]
            if not company_products.empty:
                product_row = company_products.iloc[0]
                if 'ProductType_Fk_Id' in product_row and not pd.isna(product_row['ProductType_Fk_Id']):
                    product_type = self._resolve_foreign_key(
                        'CompanyProducts', 'ProductType_Fk_Id',
                        product_row['ProductType_Fk_Id'], 'ProductTypeMaster', 'Id', 'ProductTypeName'
                    )
                    company_record['product_type'] = product_type
                if 'DefencePlatform_Fk_Id' in product_row and not pd.isna(product_row['DefencePlatform_Fk_Id']):
                    defence_platform = self._resolve_foreign_key(
                        'CompanyProducts', 'DefencePlatform_Fk_Id',
                        product_row['DefencePlatform_Fk_Id'], 'DefencePlatformMaster', 'Id', 'Name_of_Defence_Platform'
                    )
                    company_record['defence_platform'] = defence_platform
                if 'PTAType_Fk_Id' in product_row and not pd.isna(product_row['PTAType_Fk_Id']):
                    pta_type = self._resolve_foreign_key(
                        'CompanyProducts', 'PTAType_Fk_Id',
                        product_row['PTAType_Fk_Id'], 'PlatformTechAreaMaster', 'Id', 'PTAName'
                    )
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
        try:
            if 'CompanyRDFacility' not in self.tables:
                return
            rd_df = self.tables['CompanyRDFacility']
            company_rd = rd_df[rd_df['CompanyMaster_FK_ID'] == company_id]
            if not company_rd.empty:
                rd_row = company_rd.iloc[0]
                if 'RDCategory_Fk_ID' in rd_row and not pd.isna(rd_row['RDCategory_Fk_ID']):
                    rd_category = self._resolve_foreign_key(
                        'CompanyRDFacility', 'RDCategory_Fk_ID',
                        rd_row['RDCategory_Fk_ID'], 'RDCategoryMaster', 'Id', 'RDCategoryName'
                    )
                    company_record['rd_category'] = rd_category
                if 'RDSubCategory_Fk_Id' in rd_row and not pd.isna(rd_row['RDSubCategory_Fk_Id']):
                    rd_subcategory = self._resolve_foreign_key(
                        'CompanyRDFacility', 'RDSubCategory_Fk_Id',
                        rd_row['RDSubCategory_Fk_Id'], 'RDSubCategoryMaster', 'Id', 'RDSubCategoryName'
                    )
                    company_record['rd_subcategory'] = rd_subcategory
                company_record.update({
                    'rd_details': str(rd_row.get('RD_Details', '')),
                    'rd_nabl_accredited': str(rd_row.get('IsNabIAccredited', '')),
                })
        except Exception as e:
            logger.debug(f"Failed to add R&D facilities for company {company_id}: {e}")

    def _add_test_facilities(self, company_record: Dict, company_id: int):
        try:
            if 'CompanyTestFacility' not in self.tables:
                return
            test_df = self.tables['CompanyTestFacility']
            company_test = test_df[test_df['CompanyMaster_FK_ID'] == company_id]
            if not company_test.empty:
                test_row = company_test.iloc[0]
                if 'TestFacilityCategory_Fk_Id' in test_row and not pd.isna(test_row['TestFacilityCategory_Fk_Id']):
                    test_category = self._resolve_foreign_key(
                        'CompanyTestFacility', 'TestFacilityCategory_Fk_Id',
                        test_row['TestFacilityCategory_Fk_Id'], 'TestFacilityCategoryMaster', 'Id', 'CategoryName'
                    )
                    company_record['test_category'] = test_category
                if 'TestFacilitySubCategory_Fk_id' in test_row and not pd.isna(test_row['TestFacilitySubCategory_Fk_id']):
                    test_subcategory = self._resolve_foreign_key(
                        'CompanyTestFacility', 'TestFacilitySubCategory_Fk_id',
                        test_row['TestFacilitySubCategory_Fk_id'], 'TestFacilitySubCategoryMaster', 'Id', 'SubCategoryName'
                    )
                    company_record['test_subcategory'] = test_subcategory
                company_record.update({
                    'test_details': str(test_row.get('TestDetails', '')),
                    'test_nabl_accredited': str(test_row.get('IsNabIAccredited', '')),
                })
        except Exception as e:
            logger.debug(f"Failed to add test facilities for company {company_id}: {e}")

    def _add_turnover(self, company_record: Dict, company_id: int):
        try:
            if 'CompanyTurnOver' not in self.tables:
                return
            turnover_df = self.tables['CompanyTurnOver']
            company_turnover = turnover_df[turnover_df['Company_FK_Id'] == company_id]
            if not company_turnover.empty:
                turnover_row = company_turnover.iloc[0]
                company_record.update({
                    'turnover_amount': str(turnover_row.get('Amount', '')),
                    'turnover_year': str(turnover_row.get('YearId', '')),
                })
        except Exception as e:
            logger.debug(f"Failed to add turnover for company {company_id}: {e}")

    # ---------------------------- File creation ----------------------------

    def create_integrated_file(self) -> bool:
        """Create the integrated search file from flat CSV files."""
        try:
            logger.info("🚀 Starting integrated search file creation from flat files...")

            if 'CompanyMaster' not in self.tables:
                logger.error("CompanyMaster table not found!")
                return False

            company_df = self.tables['CompanyMaster']
            total_companies = len(company_df)
            logger.info(f"Processing {total_companies} companies from CompanyMaster...")

            integrated_data: List[Dict[str, Any]] = []
            processed_count = 0
            error_count = 0

            for index, company_row in company_df.iterrows():
                try:
                    company_record = self._create_company_record(company_row)
                    integrated_data.append(company_record)

                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count}/{total_companies} companies...")
                except Exception as e:
                    logger.error(f"Failed to process company at index {index}: {e}")
                    error_count += 1
                    continue

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

            output_data = {
                'metadata': metadata,
                'companies': integrated_data
            }

            logger.info(f"Writing integrated file to: {self.output_path}")
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            file_size_mb = self.output_path.stat().st_size / (1024 * 1024)

            logger.info("✅ Integrated search file created successfully!")
            logger.info(f"   📁 File: {self.output_path}")
            logger.info(f"   📊 Size: {file_size_mb:.2f} MB")
            logger.info(f"   🏢 Companies: {len(integrated_data)}")
            logger.info(f"   ✅ Processed: {processed_count}")
            logger.info(f"   ❌ Errors: {error_count}")
            logger.info(f"   📋 Tables Used: {len(self.tables)}")

            if integrated_data:
                sample_company = integrated_data[0]
                logger.info(f"   📝 Sample Company: {sample_company.get('company_name', 'Unknown')} "
                            f"({sample_company.get('company_ref_no', 'Unknown')})")
                non_empty_fields = [k for k, v in sample_company.items() if v and str(v).strip()]
                logger.info(f"   🔍 Sample Fields ({len(non_empty_fields)}): "
                            f"{', '.join(non_empty_fields[:10])}{'...' if len(non_empty_fields) > 10 else ''}")

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
    raise SystemExit(main())
