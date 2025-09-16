-- RDCategoryMaster
INSERT INTO RDCategoryMaster (RDCategoryName, IsActive) VALUES
('Power Systems', 1),
('Material Science', 1),
('Electronics', 1),
('Aerospace Research', 1),
('Telecom R&D', 1);

-- RDSubCategoryMaster
INSERT INTO RDSubCategoryMaster (RDCategory_Fk_Id, RDSubCategoryName, IsActive) VALUES
(1, 'High Voltage Labs', 1),
(2, 'Composite Testing', 1),
(3, 'Circuit Prototyping', 1),
(4, 'Satellite Propulsion', 1),
(5, '5G Signal Labs', 1);

-- CertificateTypeMaster
INSERT INTO CertificationTypeMaster (Cert_Type, IsActive) VALUES
('ISO 9001', 1),
('ISO 14001', 1),
('OHSAS 18001', 1),
('ISO 27001', 1),
('NABL Accredited', 1);

-- TestFacilityCategoryMaster
INSERT INTO TestFacilityCategoryMaster (CategoryName, IsActive) VALUES
('Electrical Testing', 1),
('Mechanical Testing', 1),
('Thermal Testing', 1),
('RF Testing', 1),
('Software QA', 1);

-- TestFacilitySubCategoryMaster
INSERT INTO TestFacilitySubCategoryMaster (TestFacility_Fk_Id, SubCategoryName, Description, IsActive) VALUES
(1, 'Insulation Testing', 'Check insulation resistance', 1),
(2, 'Tensile Strength', 'Check tensile properties', 1),
(3, 'Thermal Expansion', 'Measure material expansion', 1),
(4, 'RF Spectrum', 'Measure RF signal integrity', 1),
(5, 'Automation QA', 'Automated software testing', 1);
