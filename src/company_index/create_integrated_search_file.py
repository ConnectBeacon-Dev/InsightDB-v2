#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Integrated Company Search File (Nested Structure)

Builds one integrated JSON with, per company:
- CompanyDetails (base facts)
- ProductsAndServices -> ProductList (Products)
- QualityAndCompliance -> CertificationsList (Certifications)
- QualityAndCompliance -> TestingCapabilitiesList (TestingCapabilities)
- ResearchAndDevelopment -> RDCapabilitiesList (RDCapabilities)

Robust CSV loader (multi-encoding), FK resolution via relations.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# Optional: chardet for encoding detection
try:
    import chardet  # type: ignore
    _HAS_CHARDET = True
except Exception:
    _HAS_CHARDET = False

# ---------------------------- Logging ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("integrated_company_search")


class IntegratedSearchFileCreator:
    """Creates integrated search file from flat CSV files using table relations."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)

        # Input/Output layout (Windows-friendly; relative to repo root)
        self.input_data_path = self.base_path / "input_data" / "data_flat_file"
        self.relations_path = self.base_path / "input_data" / "table_relations" / "relations.json"
        self.processed_data_path = self.base_path / "processed_data_store" / "company_mapped_store"
        self.output_path = self.processed_data_path / "integrated_company_search.json"

        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Load relations + all CSVs
        self.relations = self._load_table_relations()
        self.tables = self._load_csv_tables()

    # ---------------------------- Robust I/O helpers ----------------------------

    def _detect_encoding(self, file_path: Path, sample_size: int = 200_000) -> Optional[str]:
        if not _HAS_CHARDET:
            return None
        try:
            with open(file_path, "rb") as f:
                raw = f.read(sample_size)
            res = chardet.detect(raw)  # {'encoding': 'utf-8', 'confidence': 0.99, ...}
            enc = (res or {}).get("encoding")
            if enc:
                logger.info(f"  ↪ chardet suggests encoding='{enc}' (conf={res.get('confidence')}) for {file_path.name}")
            return enc
        except Exception as e:
            logger.debug(f"chardet detection failed for {file_path}: {e}")
            return None

    def _read_csv_robust(self, csv_file: Path) -> pd.DataFrame:
        tried: List[str] = []
        last_err: Exception | None = None

        encodings: List[str] = []
        guess = self._detect_encoding(csv_file)
        if guess:
            encodings.append(guess)
        encodings += ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]

        for enc in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=enc)
                logger.info(f"  ✅ Loaded {csv_file.name} with encoding='{enc}': {len(df)} rows, {len(df.columns)} cols")
                return df
            except Exception as e:
                tried.append(enc)
                last_err = e
                logger.debug(f"  ↪ failed with encoding='{enc}' for {csv_file.name}: {e}")

        # Final fallback: python engine + sep sniff + latin1
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
        """Load table relations (legacy schema: list of {from_table, from_column, to_table, to_column})."""
        try:
            logger.info(f"Loading table relations from: {self.relations_path}")
            with open(self.relations_path, 'r', encoding='utf-8') as f:
                relations = json.load(f)
            # relations may be a list, or {edges:[...]} – normalize to flat list with legacy keys
            if isinstance(relations, dict) and "edges" in relations:
                edges = relations.get("edges", [])
                flat = []
                for e in edges:
                    if not isinstance(e, dict): 
                        continue
                    if "from" in e and "to" in e:
                        fobj, tobj = e["from"], e["to"]
                        flat.append({
                            "from_table": fobj.get("table"),
                            "from_column": fobj.get("field"),
                            "to_table": tobj.get("table"),
                            "to_column": tobj.get("field"),
                        })
                relations = flat
            logger.info(f"✅ Loaded {len(relations)} table relations")
            return relations
        except Exception as e:
            logger.error(f"Failed to load table relations: {e}")
            return []

    def _load_csv_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV tables into memory."""
        logger.info("Loading CSV tables...")
        tables: Dict[str, pd.DataFrame] = {}

        csv_files = list(self.input_data_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            try:
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
        target_column: Optional[str] = None,
    ) -> str:
        """Resolve foreign key to a readable value (best-effort)."""
        if pd.isna(from_value) or from_value == '' or from_value is None:
            return ""

        try:
            if to_table not in self.tables:
                return str(from_value)

            target_df = self.tables[to_table]
            # exact match
            if to_column not in target_df.columns:
                return str(from_value)

            matching_rows = target_df[target_df[to_column] == from_value]
            if matching_rows.empty:
                return str(from_value)

            if target_column and target_column in matching_rows.columns:
                return str(matching_rows.iloc[0][target_column])

            # fallback: pick a name-ish column
            name_cols = [c for c in matching_rows.columns if 'name' in c.lower() or 'type' in c.lower()]
            if name_cols:
                return str(matching_rows.iloc[0][name_cols[0]])

            return str(from_value)

        except Exception as e:
            logger.debug(f"FK resolution failed for {from_table}.{from_column} -> {to_table}.{to_column}: {e}")
            return str(from_value) if from_value else ""

    def _get_fk_map(self, table_name: str) -> Dict[str, Dict]:
        """Map FK columns for a given table (legacy relations schema)."""
        fk_mappings: Dict[str, Dict] = {}
        for rel in self.relations:
            if rel.get('from_table') == table_name:
                fk_mappings[rel['from_column']] = {
                    'to_table': rel['to_table'],
                    'to_column': rel['to_column']
                }
        return fk_mappings

    # ---------------------------- Section builders ----------------------------

    def _build_company_details(self, row: pd.Series) -> Dict[str, Any]:
        """CompanyDetails section (base)."""
        company_id = row.get('Id')
        details: Dict[str, Any] = {
            'company_ref_no': str(row.get('CompanyRefNo', '') or ''),
            'company_id': int(company_id) if str(company_id).isdigit() else 0,
            'company_name': str(row.get('CompanyName', '') or ''),
            'cin_number': str(row.get('CINNumber', '') or ''),
            'pan': str(row.get('Pan', '') or ''),
            'registration_date': str(row.get('CompanyRegistrationDate', '') or ''),
            'company_status': str(row.get('CompanyStatus', '') or ''),
            'company_class': str(row.get('CompanyClass', '') or ''),
            'listing_status': str(row.get('ListingStatus', '') or ''),
            'company_category': str(row.get('CompanyCategory', '') or ''),
            'company_subcategory': str(row.get('CompanySubCategory', '') or ''),
            'industrial_classification': str(row.get('CompanyIndustrialClassification', '') or ''),
            'other_expertise': str(row.get('OtherCompanyCoreExpertise', '') or ''),
            'other_industry_domain': str(row.get('OtherCompIndDomain', '') or ''),
            'other_industry_subdomain': str(row.get('OtherCompIndSubDomain', '') or ''),
            # Contacts
            'address': str(row.get('Address', '') or ''),
            'city': str(row.get('CityName', '') or ''),
            'district': str(row.get('District', '') or ''),
            'state': str(row.get('State', '') or ''),
            'pincode': str(row.get('PinCode', '') or ''),
            'email': str(row.get('EmailId', '') or ''),
            'poc_email': str(row.get('POC_Email', '') or ''),
            'phone': str(row.get('Phone', '') or ''),
            'website': str(row.get('Website', '') or ''),
        }

        # Selected FK enrichments (best-effort)
        # Country_Fk_id -> CountryMaster.CountryName
        if 'Country_Fk_id' in row and pd.notna(row['Country_Fk_id']):
            details['country'] = self._resolve_foreign_key(
                'CompanyMaster', 'Country_Fk_id',
                row['Country_Fk_id'], 'CountryMaster', 'Id', 'CountryName'
            )
        else:
            details['country'] = ''

        # CompanyScale_Fk_Id -> ScaleMaster.CompanyScale
        if 'CompanyScale_Fk_Id' in row and pd.notna(row['CompanyScale_Fk_Id']):
            details['company_scale'] = self._resolve_foreign_key(
                'CompanyMaster', 'CompanyScale_Fk_Id',
                row['CompanyScale_Fk_Id'], 'ScaleMaster', 'Id', 'CompanyScale'
            )
        else:
            details['company_scale'] = ''

        # CompanyType_Fk_Id -> OrganisationTypeMaster.Organization_Type
        if 'CompanyType_Fk_Id' in row and pd.notna(row['CompanyType_Fk_Id']):
            details['organization_type'] = self._resolve_foreign_key(
                'CompanyMaster', 'CompanyType_Fk_Id',
                row['CompanyType_Fk_Id'], 'OrganisationTypeMaster', 'Id', 'Organization_Type'
            )
        else:
            details['organization_type'] = ''

        # CompanyCoreExpertise_Fk_Id -> CompanyCoreExpertiseMaster.CoreExpertiseName
        if 'CompanyCoreExpertise_Fk_Id' in row and pd.notna(row['CompanyCoreExpertise_Fk_Id']):
            details['core_expertise'] = self._resolve_foreign_key(
                'CompanyMaster', 'CompanyCoreExpertise_Fk_Id',
                row['CompanyCoreExpertise_Fk_Id'], 'CompanyCoreExpertiseMaster', 'Id', 'CoreExpertiseName'
            )
        else:
            details['core_expertise'] = ''

        # IndustryDomain_Fk_Id -> IndustryDomainMaster.IndustryDomainType
        if 'IndustryDomain_Fk_Id' in row and pd.notna(row['IndustryDomain_Fk_Id']):
            details['industry_domain'] = self._resolve_foreign_key(
                'CompanyMaster', 'IndustryDomain_Fk_Id',
                row['IndustryDomain_Fk_Id'], 'IndustryDomainMaster', 'Id', 'IndustryDomainType'
            )
        else:
            details['industry_domain'] = ''

        # IndustrySubDomain_Fk_Id -> IndustrySubdomainType.IndustrySubDomainName
        if 'IndustrySubDomain_Fk_Id' in row and pd.notna(row['IndustrySubDomain_Fk_Id']):
            details['industry_subdomain'] = self._resolve_foreign_key(
                'CompanyMaster', 'IndustrySubDomain_Fk_Id',
                row['IndustrySubDomain_Fk_Id'], 'IndustrySubdomainType', 'Id', 'IndustrySubDomainName'
            )
        else:
            details['industry_subdomain'] = ''

        return details

    def _build_products_list(self, company_id: Any) -> List[Dict[str, Any]]:
        """ProductsAndServices -> ProductList (Products)."""
        out: List[Dict[str, Any]] = []
        try:
            if 'CompanyProducts' not in self.tables:
                return out

            df = self.tables['CompanyProducts']
            rows = df[df.get('CompanyMaster_FK_Id') == company_id]
            for _, r in rows.iterrows():
                item = {
                    'product_name': str(r.get('ProductName', '') or ''),
                    'product_description': str(r.get('ProductDesc', '') or ''),
                    'hsn_code': str(r.get('HSNCode', '') or ''),
                    'nsn_number': str(r.get('NSNNumber', '') or ''),
                    'items_exported': str(r.get('ItemExported', '') or ''),
                    'product_certificates': str(r.get('ProductCertificateDet', '') or ''),
                    'salient_features': str(r.get('SalientFeature', '') or ''),
                    'annual_production_capacity': str(r.get('AnnualProductionCapacity', '') or ''),
                    'future_expansion': str(r.get('FutureExpansion', '') or ''),
                }
                # ProductType
                if pd.notna(r.get('ProductType_Fk_Id')):
                    item['product_type'] = self._resolve_foreign_key(
                        'CompanyProducts', 'ProductType_Fk_Id', r['ProductType_Fk_Id'],
                        'ProductTypeMaster', 'Id', 'ProductTypeName'
                    )
                else:
                    item['product_type'] = ''
                # DefencePlatform
                if pd.notna(r.get('DefencePlatform_Fk_Id')):
                    item['defence_platform'] = self._resolve_foreign_key(
                        'CompanyProducts', 'DefencePlatform_Fk_Id', r['DefencePlatform_Fk_Id'],
                        'DefencePlatformMaster', 'Id', 'Name_of_Defence_Platform'
                    )
                else:
                    item['defence_platform'] = ''
                # PlatformTechArea
                if pd.notna(r.get('PTAType_Fk_Id')):
                    item['platform_tech_area'] = self._resolve_foreign_key(
                        'CompanyProducts', 'PTAType_Fk_Id', r['PTAType_Fk_Id'],
                        'PlatformTechAreaMaster', 'Id', 'PTAName'
                    )
                else:
                    item['platform_tech_area'] = ''
                out.append(item)
        except Exception as e:
            logger.debug(f"Products build failed for company {company_id}: {e}")
        return out

    def _build_certifications_list(self, company_id: Any) -> List[Dict[str, Any]]:
        """QualityAndCompliance -> CertificationsList (Certifications)."""
        out: List[Dict[str, Any]] = []
        try:
            if 'CompanyCertificationDetail' not in self.tables:
                return out

            df = self.tables['CompanyCertificationDetail']
            rows = df[df.get('CompanyMaster_Fk_Id') == company_id]
            for _, r in rows.iterrows():
                item = {
                    'certification_detail': str(r.get('Certification_Type', '') or ''),
                    'other_certification_type': str(r.get('OtherCertification_Type', '') or ''),
                    'certificate_number': str(r.get('Certificate_No', '') or ''),
                    'certificate_start_date': str(r.get('Certificate_StartDate', '') or ''),
                    'certificate_end_date': str(r.get('Certificate_EndDate', '') or ''),
                }
                # CertificationType
                if pd.notna(r.get('CertificateType_Fk_Id')):
                    item['certification_type_master'] = self._resolve_foreign_key(
                        'CompanyCertificationDetail', 'CertificateType_Fk_Id', r['CertificateType_Fk_Id'],
                        'CertificationTypeMaster', 'Id', 'Cert_Type'
                    )
                else:
                    item['certification_type_master'] = ''
                out.append(item)
        except Exception as e:
            logger.debug(f"Certifications build failed for company {company_id}: {e}")
        return out

    def _build_testing_capabilities_list(self, company_id: Any) -> List[Dict[str, Any]]:
        """QualityAndCompliance -> TestingCapabilitiesList (TestingCapabilities)."""
        out: List[Dict[str, Any]] = []
        try:
            if 'CompanyTestFacility' not in self.tables:
                return out

            df = self.tables['CompanyTestFacility']
            rows = df[df.get('CompanyMaster_FK_ID') == company_id]
            for _, r in rows.iterrows():
                item = {
                    'test_details': str(r.get('TestDetails', '') or ''),
                    'test_nabl_accredited': str(r.get('IsNabIAccredited', '') or ''),
                }
                # Category
                if pd.notna(r.get('TestFacilityCategory_Fk_Id')):
                    item['test_category'] = self._resolve_foreign_key(
                        'CompanyTestFacility', 'TestFacilityCategory_Fk_Id', r['TestFacilityCategory_Fk_Id'],
                        'TestFacilityCategoryMaster', 'Id', 'CategoryName'
                    )
                else:
                    item['test_category'] = ''
                # SubCategory
                if pd.notna(r.get('TestFacilitySubCategory_Fk_id')):
                    item['test_subcategory'] = self._resolve_foreign_key(
                        'CompanyTestFacility', 'TestFacilitySubCategory_Fk_id', r['TestFacilitySubCategory_Fk_id'],
                        'TestFacilitySubCategoryMaster', 'Id', 'SubCategoryName'
                    )
                    # Optional description
                    item['test_subcategory_description'] = self._resolve_foreign_key(
                        'CompanyTestFacility', 'TestFacilitySubCategory_Fk_id', r['TestFacilitySubCategory_Fk_id'],
                        'TestFacilitySubCategoryMaster', 'Id', 'Description'
                    )
                else:
                    item['test_subcategory'] = ''
                    item['test_subcategory_description'] = ''
                out.append(item)
        except Exception as e:
            logger.debug(f"Testing capabilities build failed for company {company_id}: {e}")
        return out

    def _build_rd_capabilities_list(self, company_id: Any) -> List[Dict[str, Any]]:
        """ResearchAndDevelopment -> RDCapabilitiesList (RDCapabilities)."""
        out: List[Dict[str, Any]] = []
        try:
            if 'CompanyRDFacility' not in self.tables:
                return out

            df = self.tables['CompanyRDFacility']
            rows = df[df.get('CompanyMaster_FK_ID') == company_id]
            for _, r in rows.iterrows():
                item = {
                    'rd_details': str(r.get('RD_Details', '') or ''),
                    'rd_nabl_accredited': str(r.get('IsNabIAccredited', '') or ''),
                }
                # Category
                if pd.notna(r.get('RDCategory_Fk_ID')):
                    item['rd_category'] = self._resolve_foreign_key(
                        'CompanyRDFacility', 'RDCategory_Fk_ID', r['RDCategory_Fk_ID'],
                        'RDCategoryMaster', 'Id', 'RDCategoryName'
                    )
                else:
                    item['rd_category'] = ''
                # SubCategory
                if pd.notna(r.get('RDSubCategory_Fk_Id')):
                    item['rd_subcategory'] = self._resolve_foreign_key(
                        'CompanyRDFacility', 'RDSubCategory_Fk_Id', r['RDSubCategory_Fk_Id'],
                        'RDSubCategoryMaster', 'Id', 'RDSubCategoryName'
                    )
                else:
                    item['rd_subcategory'] = ''
                out.append(item)
        except Exception as e:
            logger.debug(f"R&D capabilities build failed for company {company_id}: {e}")
        return out

    # (Optional) Finances or other sections can go here as needed
    def _build_financials(self, company_id: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            if 'CompanyTurnOver' not in self.tables:
                return out
            df = self.tables['CompanyTurnOver']
            rows = df[df.get('Company_FK_Id') == company_id]
            for _, r in rows.iterrows():
                out.append({
                    'amount': str(r.get('Amount', '') or ''),
                    'year_id': str(r.get('YearId', '') or ''),
                })
        except Exception as e:
            logger.debug(f"Financials build failed for company {company_id}: {e}")
        return out

    # ---------------------------- Record assembly ----------------------------

    def _create_company_record(self, row: pd.Series) -> Dict[str, Any]:
        """Compose the full nested object per company."""
        try:
            company_id = row.get('Id')

            company_details = self._build_company_details(row)
            products_list = self._build_products_list(company_id)
            certs_list = self._build_certifications_list(company_id)
            testing_list = self._build_testing_capabilities_list(company_id)
            rd_list = self._build_rd_capabilities_list(company_id)
            # financials = self._build_financials(company_id)  # add to CompanyDetails if needed

            # Compose nested object exactly per your requested structure
            company_obj: Dict[str, Any] = {
                "CompanyDetails": company_details,
                "ProductsAndServices": {
                    "ProductList": products_list
                },
                "QualityAndCompliance": {
                    "CertificationsList": certs_list,
                    "TestingCapabilitiesList": testing_list
                },
                "ResearchAndDevelopment": {
                    "RDCapabilitiesList": rd_list
                }
            }

            return company_obj

        except Exception as e:
            logger.error(f"Failed to create company record for ID {row.get('Id', 'unknown')}: {e}")
            return {
                "CompanyDetails": {
                    "company_ref_no": str(row.get('CompanyRefNo', '') or ''),
                    "company_id": int(row.get('Id', 0)) if str(row.get('Id', '0')).isdigit() else 0,
                    "company_name": str(row.get('CompanyName', 'Unknown'))
                },
                "ProductsAndServices": { "ProductList": [] },
                "QualityAndCompliance": {
                    "CertificationsList": [],
                    "TestingCapabilitiesList": []
                },
                "ResearchAndDevelopment": { "RDCapabilitiesList": [] },
                "error": str(e)
            }

    # ---------------------------- Top-level creation ----------------------------

    def create_integrated_file(self) -> bool:
        """Create the integrated search file with nested structure."""
        try:
            logger.info("🚀 Creating integrated company search file (nested)...")

            if 'CompanyMaster' not in self.tables:
                logger.error("CompanyMaster table not found!")
                return False

            company_df = self.tables['CompanyMaster']
            total = len(company_df)
            logger.info(f"Processing {total} companies from CompanyMaster...")

            companies: List[Dict[str, Any]] = []
            processed = 0
            errors = 0

            for idx, row in company_df.iterrows():
                try:
                    companies.append(self._create_company_record(row))
                    processed += 1
                    if processed % 10 == 0:
                        logger.info(f"Processed {processed}/{total} companies...")
                except Exception as e:
                    logger.error(f"Failed at index {idx}: {e}")
                    errors += 1

            metadata = {
                "created_at": datetime.now().isoformat(),
                "total_companies": len(companies),
                "processed_successfully": processed,
                "errors": errors,
                "source_files": {
                    "flat_files_dir": str(self.input_data_path),
                    "relations_file": str(self.relations_path),
                    "tables_loaded": list(self.tables.keys())
                },
                "schema_version": "3.0",
                "description": (
                    "Integrated company search file with CompanyDetails, "
                    "ProductsAndServices/ProductList, QualityAndCompliance/{CertificationsList,TestingCapabilitiesList}, "
                    "ResearchAndDevelopment/RDCapabilitiesList."
                )
            }

            output = { "metadata": metadata, "companies": companies }

            logger.info(f"Writing integrated file to: {self.output_path}")
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            size_mb = self.output_path.stat().st_size / (1024 * 1024)
            logger.info("✅ Integrated file created successfully!")
            logger.info(f"   File: {self.output_path}")
            logger.info(f"   Size: {size_mb:.2f} MB")
            logger.info(f"   Companies: {len(companies)}")
            logger.info(f"   Processed: {processed}")
            logger.info(f"   Errors: {errors}")
            logger.info(f"   Tables Used: {len(self.tables)}")

            if companies:
                sample = companies[0]
                # Show a quick preview of keys present
                logger.info(f"   Sample top-level keys: {list(sample.keys())}")

            return True

        except Exception as e:
            logger.error(f"Failed to create integrated search file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def main():
    creator = IntegratedSearchFileCreator()
    ok = creator.create_integrated_file()

    if ok:
        print(" Integrated search file created successfully (nested structure)!")
    else:
        print(" Failed to create integrated search file")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
