#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test case to validate generated CSV files from generate_grouped_csvs_with_data.py

This script validates:
1. File count correctness
2. Content synchronization with input files
3. Data integrity and consistency
4. Column mapping accuracy
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import sys

from src.load_config import load_config, get_processing_params, get_logger

# Setup logging
logger = get_logger()
logger.setLevel(logging.INFO)

class CSVValidationError(Exception):
    """Custom exception for CSV validation errors"""
    pass

class GeneratedCSVValidator:
    """Validator class for generated CSV files"""
    
    def __init__(self, config_path: str = None):
        """Initialize validator with configuration"""
        # load_config() returns a tuple (config, logger), so we need to unpack it
        if config_path:
            self.config, _ = load_config(config_path)
        else:
            self.config, _ = load_config()
        self.params = get_processing_params(self.config)
        
        self.domain_mapping_file = self.params["domain_mapping"]
        self.relations_file = self.params["table_relations"]
        self.output_dir = self.params["outdir"]
        self.input_tables_dir = self.params["tables_dir"]
        
        # Load configurations
        self.domain_mapping = self._load_json(self.domain_mapping_file)
        self.relations = self._load_json(self.relations_file)
        
        # Validation results
        self.validation_results = {
            "file_count_validation": {},
            "content_validation": {},
            "data_integrity": {},
            "errors": [],
            "warnings": []
        }
        
        logger.info("CSV Validator initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Input tables directory: {self.input_tables_dir}")
    
    def _load_json(self, file_path: Path) -> Any:
        """Load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise
    
    def _load_csv_safe(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with error handling"""
        try:
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False, na_values=[])
            logger.debug(f"Loaded CSV: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            raise CSVValidationError(f"Cannot load CSV file {file_path}: {e}")
    
    def validate_file_count(self) -> bool:
        """Validate that the correct number of files were generated"""
        logger.info("=== VALIDATING FILE COUNT ===")
        
        # Count expected files from domain mapping
        expected_files = []
        for group, subgroups in self.domain_mapping.items():
            if isinstance(subgroups, dict):
                for subgroup in subgroups.keys():
                    safe_group = self._safe_name(group)
                    safe_subgroup = self._safe_name(subgroup)
                    expected_files.append(f"{safe_group}.{safe_subgroup}.csv")
        
        expected_count = len(expected_files)
        logger.info(f"Expected {expected_count} CSV files based on domain mapping")
        
        # Count actual generated files
        if not self.output_dir.exists():
            error_msg = f"Output directory does not exist: {self.output_dir}"
            logger.error(error_msg)
            self.validation_results["errors"].append(error_msg)
            return False
        
        actual_files = list(self.output_dir.glob("*.csv"))
        # Exclude manifest file from count
        actual_data_files = [f for f in actual_files if f.name != "_index.csv"]
        actual_count = len(actual_data_files)
        
        logger.info(f"Found {actual_count} actual CSV files in output directory")
        logger.info(f"Found files: {[f.name for f in actual_data_files]}")
        
        # Validate counts match
        count_match = expected_count == actual_count
        self.validation_results["file_count_validation"] = {
            "expected_count": expected_count,
            "actual_count": actual_count,
            "count_match": count_match,
            "expected_files": expected_files,
            "actual_files": [f.name for f in actual_data_files],
            "missing_files": list(set(expected_files) - set([f.name for f in actual_data_files])),
            "extra_files": list(set([f.name for f in actual_data_files]) - set(expected_files))
        }
        
        if count_match:
            logger.info("✅ File count validation PASSED")
        else:
            logger.error("❌ File count validation FAILED")
            if self.validation_results["file_count_validation"]["missing_files"]:
                logger.error(f"Missing files: {self.validation_results['file_count_validation']['missing_files']}")
            if self.validation_results["file_count_validation"]["extra_files"]:
                logger.error(f"Extra files: {self.validation_results['file_count_validation']['extra_files']}")
        
        return count_match
    
    def validate_manifest_file(self) -> bool:
        """Validate the manifest/index file"""
        logger.info("=== VALIDATING MANIFEST FILE ===")
        
        manifest_path = self.output_dir / "_index.csv"
        if not manifest_path.exists():
            error_msg = "Manifest file (_index.csv) does not exist"
            logger.error(error_msg)
            self.validation_results["errors"].append(error_msg)
            return False
        
        try:
            manifest_df = self._load_csv_safe(manifest_path)
            logger.info(f"Manifest file loaded with {len(manifest_df)} entries")
            
            # Validate manifest columns
            expected_columns = ["csv_name", "group", "subgroup", "rows", "columns", "first_cols"]
            actual_columns = list(manifest_df.columns)
            
            if set(expected_columns) != set(actual_columns):
                error_msg = f"Manifest columns mismatch. Expected: {expected_columns}, Got: {actual_columns}"
                logger.error(error_msg)
                self.validation_results["errors"].append(error_msg)
                return False
            
            # Validate each entry in manifest corresponds to actual file
            for _, row in manifest_df.iterrows():
                csv_file_path = self.output_dir / row["csv_name"]
                if not csv_file_path.exists():
                    error_msg = f"File listed in manifest does not exist: {row['csv_name']}"
                    logger.error(error_msg)
                    self.validation_results["errors"].append(error_msg)
                    return False
                
                # Validate file metadata
                actual_df = self._load_csv_safe(csv_file_path)
                if len(actual_df) != int(row["rows"]):
                    error_msg = f"Row count mismatch for {row['csv_name']}: manifest={row['rows']}, actual={len(actual_df)}"
                    logger.error(error_msg)
                    self.validation_results["errors"].append(error_msg)
                    return False
                
                if len(actual_df.columns) != int(row["columns"]):
                    error_msg = f"Column count mismatch for {row['csv_name']}: manifest={row['columns']}, actual={len(actual_df.columns)}"
                    logger.error(error_msg)
                    self.validation_results["errors"].append(error_msg)
                    return False
            
            logger.info("✅ Manifest file validation PASSED")
            return True
            
        except Exception as e:
            error_msg = f"Error validating manifest file: {e}"
            logger.error(error_msg)
            self.validation_results["errors"].append(error_msg)
            return False
    
    def validate_content_integrity(self) -> bool:
        """Validate content integrity of generated files"""
        logger.info("=== VALIDATING CONTENT INTEGRITY ===")
        
        all_valid = True
        
        for group, subgroups in self.domain_mapping.items():
            if not isinstance(subgroups, dict):
                continue
                
            for subgroup, fields in subgroups.items():
                logger.info(f"Validating content for {group}.{subgroup}")
                
                # Load generated file
                safe_group = self._safe_name(group)
                safe_subgroup = self._safe_name(subgroup)
                generated_file = self.output_dir / f"{safe_group}.{safe_subgroup}.csv"
                
                if not generated_file.exists():
                    error_msg = f"Generated file does not exist: {generated_file}"
                    logger.error(error_msg)
                    self.validation_results["errors"].append(error_msg)
                    all_valid = False
                    continue
                
                try:
                    generated_df = self._load_csv_safe(generated_file)
                    validation_result = self._validate_single_file_content(
                        generated_df, group, subgroup, fields
                    )
                    
                    self.validation_results["content_validation"][f"{group}.{subgroup}"] = validation_result
                    
                    if not validation_result["valid"]:
                        all_valid = False
                        logger.error(f"Content validation failed for {group}.{subgroup}")
                        for error in validation_result["errors"]:
                            logger.error(f"  - {error}")
                    else:
                        logger.info(f"✅ Content validation passed for {group}.{subgroup}")
                        
                except Exception as e:
                    error_msg = f"Error validating content for {group}.{subgroup}: {e}"
                    logger.error(error_msg)
                    self.validation_results["errors"].append(error_msg)
                    all_valid = False
        
        if all_valid:
            logger.info("✅ Content integrity validation PASSED")
        else:
            logger.error("❌ Content integrity validation FAILED")
        
        return all_valid
    
    def _validate_single_file_content(self, df: pd.DataFrame, group: str, subgroup: str, fields: List[Dict]) -> Dict:
        """Validate content of a single generated file"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns)
            }
        }
        
        # Validate required columns exist
        required_columns = ["CompanyRefNo", "CompanyNumber"]
        for req_col in required_columns:
            if req_col not in df.columns:
                validation_result["errors"].append(f"Required column missing: {req_col}")
                validation_result["valid"] = False
        
        # Validate mapped fields exist
        for field_mapping in (fields or []):
            if not isinstance(field_mapping, dict):
                continue
            
            table = field_mapping.get("table")
            field = field_mapping.get("field")
            
            if not table or not field:
                continue
            
            # Check for qualified column name
            qualified_name = f"{table}.{field}"
            if qualified_name not in df.columns and field not in df.columns:
                validation_result["errors"].append(f"Mapped field not found: {qualified_name} or {field}")
                validation_result["valid"] = False
        
        # Validate data consistency
        if "CompanyRefNo" in df.columns:
            null_count = df["CompanyRefNo"].isna().sum()
            if null_count > 0:
                validation_result["warnings"].append(f"Found {null_count} null values in CompanyRefNo")
        
        # Check for duplicate company references
        if "CompanyRefNo" in df.columns:
            duplicate_count = df["CompanyRefNo"].duplicated().sum()
            if duplicate_count > 0:
                validation_result["warnings"].append(f"Found {duplicate_count} duplicate CompanyRefNo values")
        
        return validation_result
    
    def validate_data_relationships(self) -> bool:
        """Validate data relationships and joins are correct"""
        logger.info("=== VALIDATING DATA RELATIONSHIPS ===")
        
        # Load base company table for reference
        company_csv_files = list(self.input_tables_dir.glob("*CompanyMaster*.csv"))
        if not company_csv_files:
            error_msg = "CompanyMaster CSV file not found in input directory"
            logger.error(error_msg)
            self.validation_results["errors"].append(error_msg)
            return False
        
        company_df = self._load_csv_safe(company_csv_files[0])
        company_ref_col = None
        
        # Find the company reference column
        for col in company_df.columns:
            if "CompanyRefNo" in col or col.strip() == "CompanyRefNo":
                company_ref_col = col.strip()
                break
        
        if not company_ref_col:
            error_msg = "CompanyRefNo column not found in CompanyMaster"
            logger.error(error_msg)
            self.validation_results["errors"].append(error_msg)
            return False
        
        base_company_refs = set(company_df[company_ref_col].astype(str))
        logger.info(f"Found {len(base_company_refs)} companies in base CompanyMaster table")
        
        all_valid = True
        
        # Validate each generated file has valid company references
        for csv_file in self.output_dir.glob("*.csv"):
            if csv_file.name == "_index.csv":
                continue
            
            try:
                generated_df = self._load_csv_safe(csv_file)
                
                if "CompanyRefNo" not in generated_df.columns:
                    continue
                
                file_company_refs = set(generated_df["CompanyRefNo"].astype(str))
                invalid_refs = file_company_refs - base_company_refs
                
                if invalid_refs:
                    error_msg = f"Invalid company references in {csv_file.name}: {len(invalid_refs)} invalid refs"
                    logger.error(error_msg)
                    logger.debug(f"Sample invalid refs: {list(invalid_refs)[:5]}")
                    self.validation_results["errors"].append(error_msg)
                    all_valid = False
                else:
                    logger.info(f"✅ All company references valid in {csv_file.name}")
                    
            except Exception as e:
                error_msg = f"Error validating relationships in {csv_file.name}: {e}"
                logger.error(error_msg)
                self.validation_results["errors"].append(error_msg)
                all_valid = False
        
        if all_valid:
            logger.info("✅ Data relationships validation PASSED")
        else:
            logger.error("❌ Data relationships validation FAILED")
        
        return all_valid
    
    def _safe_name(self, name: str) -> str:
        """Convert name to safe filename format"""
        return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("🚀 STARTING COMPREHENSIVE CSV VALIDATION")
        logger.info("=" * 60)
        
        validation_steps = [
            ("File Count", self.validate_file_count),
            ("Manifest File", self.validate_manifest_file),
            ("Content Integrity", self.validate_content_integrity),
            ("Data Relationships", self.validate_data_relationships)
        ]
        
        all_passed = True
        
        for step_name, validation_func in validation_steps:
            logger.info(f"\n📋 Running {step_name} validation...")
            try:
                result = validation_func()
                if result:
                    logger.info(f"✅ {step_name} validation PASSED")
                else:
                    logger.error(f"❌ {step_name} validation FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"💥 {step_name} validation ERROR: {e}")
                self.validation_results["errors"].append(f"{step_name} validation error: {e}")
                all_passed = False
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        if all_passed:
            logger.info("🎉 ALL VALIDATIONS PASSED! Generated files are correct.")
        else:
            logger.error("💥 VALIDATION FAILED! Issues found with generated files.")
        
        logger.info(f"Total errors: {len(self.validation_results['errors'])}")
        logger.info(f"Total warnings: {len(self.validation_results['warnings'])}")
        
        if self.validation_results["errors"]:
            logger.info("\n🚨 ERRORS:")
            for i, error in enumerate(self.validation_results["errors"], 1):
                logger.error(f"  {i}. {error}")
        
        if self.validation_results["warnings"]:
            logger.info("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.validation_results["warnings"], 1):
                logger.warning(f"  {i}. {warning}")
        
        return all_passed
    
    def generate_validation_report(self) -> str:
        """Generate detailed validation report"""
        report_lines = [
            "CSV VALIDATION REPORT",
            "=" * 50,
            f"Timestamp: {pd.Timestamp.now()}",
            f"Output Directory: {self.output_dir}",
            f"Input Directory: {self.input_tables_dir}",
            "",
            "FILE COUNT VALIDATION:",
            f"  Expected: {self.validation_results['file_count_validation'].get('expected_count', 'N/A')}",
            f"  Actual: {self.validation_results['file_count_validation'].get('actual_count', 'N/A')}",
            f"  Match: {self.validation_results['file_count_validation'].get('count_match', 'N/A')}",
            "",
            "CONTENT VALIDATION:",
        ]
        
        for file_key, validation in self.validation_results["content_validation"].items():
            report_lines.extend([
                f"  {file_key}:",
                f"    Valid: {validation['valid']}",
                f"    Rows: {validation['stats']['row_count']}",
                f"    Columns: {validation['stats']['column_count']}",
                f"    Errors: {len(validation['errors'])}",
                f"    Warnings: {len(validation['warnings'])}",
            ])
        
        report_lines.extend([
            "",
            f"TOTAL ERRORS: {len(self.validation_results['errors'])}",
            f"TOTAL WARNINGS: {len(self.validation_results['warnings'])}",
        ])
        
        return "\n".join(report_lines)


def main():
    """Main function to run validation"""
    try:
        validator = GeneratedCSVValidator()
        success = validator.run_full_validation()
        
        # Generate and save report
        report = validator.generate_validation_report()
        report_file = validator.output_dir / "validation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 Validation report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        logger.exception("Full exception details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
