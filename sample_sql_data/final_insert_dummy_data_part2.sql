-- CompanyRDFacility
INSERT INTO CompanyRDFacility (CompanyMaster_FK_ID, CompanyRefNo, RDRefNo, RDCategory_Fk_ID, RDSubCategory_Fk_Id, RD_Details, IsNabIAccredited, IsActive, Final_Submit, CreatedBy, CreatedDate, CreatedIP, UpdatedBy, UpdateDate, UpdatedIP) VALUES
(1, 'CMP001', 'RD001', 1, 1, 'High voltage transformer R&D facility', 1, 1, 1, 'admin', GETDATE(), '127.0.0.1', 'admin', GETDATE(), '127.0.0.1'),
(2, 'CMP002', 'RD002', 2, 2, 'Material fatigue testing lab', 1, 1, 1, 'admin', GETDATE(), '127.0.0.1', 'admin', GETDATE(), '127.0.0.1');

-- CompanyCertificationDetail
INSERT INTO CompanyCertificationDetail (CompanyMaster_Fk_Id, CompanyRefNo, CertificateRefNo, Certification_Type, OtherCertification_Type, Certificate_No, Certificate_StartDate, Certificate_EndDate, Final_Submit, IsActive, CreatedBy, CreatedDate, CreatedIP, UpdatedBy, UpdatedDate, UpdatedIP, CertificateType_Fk_Id) VALUES
(1, 'CMP001', 'CERT001', 'ISO 9001', NULL, 'ISO9001-001', '2023-01-01', '2026-01-01', 1, 1, 'admin', GETDATE(), '127.0.0.1', 'admin', GETDATE(), '127.0.0.1', 1),
(2, 'CMP002', 'CERT002', 'ISO 14001', NULL, 'ISO14001-002', '2023-01-01', '2026-01-01', 1, 1, 'admin', GETDATE(), '127.0.0.1', 'admin', GETDATE(), '127.0.0.1', 2);

-- CompanyTestFacility
INSERT INTO CompanyTestFacility (CompanyMaster_FK_ID, CompanyRefNo, TestRefNo, TestDetails, Final_Submit, IsActive, CreatedBy, CreatedDate, CreatedIP, UpdatedBy, UpdatedDate, UpdateIP, IsNabIAccredited, TestFacilityCategory_Fk_Id, TestFacilitySubCategory_Fk_id) VALUES
(1, 'CMP001', 'TF001', 'Routine insulation tests for transformers', 1, 1, 'admin', GETDATE(), '127.0.0.1', 'admin', GETDATE(), '127.0.0.1', 1, 1, 1),
(2, 'CMP002', 'TF002', 'Tensile testing of mechanical parts', 1, 1, 'admin', GETDATE(), '127.0.0.1', 'admin', GETDATE(), '127.0.0.1', 1, 2, 2);

-- CompanyTurnOver
INSERT INTO CompanyTurnOver (Company_FK_Id, YearId, Amount, Status, Final_Submit, CreatedBy, CreatedDate, CreatedIP, UpdatedBy, UpdatedDate, UpdateIP, IsActive) VALUES
(1, '2023-2024', '10000000', 'Audited', 1, 'admin', GETDATE(), '127.0.0.1', 'admin', GETDATE(), '127.0.0.1', 1),
(2, '2023-2024', '15000000', 'Audited', 1, 'admin', GETDATE(), '127.0.0.1', 'admin', GETDATE(), '127.0.0.1', 1);
