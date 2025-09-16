
-- Populating RD Categories and Subcategories
INSERT INTO RDCategoryMaster (RDCategoryName, IsActive) VALUES
('Power Systems', 1), ('Aerospace Research', 1), ('Telecom Systems', 1), ('Software Innovations', 1);

INSERT INTO RDSubCategoryMaster (RDCategory_Fk_Id, RDSubCategoryName, IsActive) VALUES
(1, 'High Voltage Lab', 1),
(2, 'Satellite Payload Testing', 1),
(3, '5G NR Lab', 1),
(4, 'AI Model Training', 1);

-- Populating Year Master
INSERT INTO YearMaster (Year, IsActive) VALUES
('2023-2024', 1), ('2024-2025', 1);

-- Populating Certification Type
INSERT INTO CertificationTypeMaster (Cert_Type, IsActive) VALUES
('ISO 9001', 1), ('ISO 27001', 1);

-- Populating Product Types
INSERT INTO ProductTypeMaster (ProductTypeName, IsActive) VALUES
('Transformer', 1), ('Radar', 1), ('5G Antenna', 1), ('AI Software', 1);

-- Populating Platform Tech Area
INSERT INTO PlatformTechAreaMaster (PTAName, IsActive) VALUES
('High Voltage Systems', 1), ('Satellite Communications', 1);

-- Populating Defence Platform Master
INSERT INTO DefencePlatformMaster (Name_of_Defence_Platform, IsActive) VALUES
('Radar Systems', 1), ('Missile Guidance', 1);

-- Example Company Master (aligned with expertise & country)
INSERT INTO CompanyMaster (CINNumber, Pan, CompanyRefNo, DPSU_Fk_Id, CompanyName, POC_Email, Phone, EmailId, Address, CityName, PinCode, Country_Fk_id, District, State, Website, Logo, CompanyScale_Fk_Id, CompanyType_Fk_Id, IndustryDomain_Fk_Id, IndustrySubDomain_Fk_Id, CompanyCoreExpertise_Fk_Id, CompanyRegistrationDate, CompanyStatus, CompanyCategory, CompanySubCategory, CompanyClass, ListingStatus, CompanyROC, CompanyIndustrialClassification, Final_Submit, IsActive, CreatedDate, CreatedBy, IPAddress)
VALUES
('CIN123', 'PAN123', 'CMP001', 'DPSU1', 'ElectroTech Pvt Ltd', 'contact@electrotech.com', '1234567890', 'support@electrotech.com', '123 Industrial Zone', 'Bangalore', '560001', 1, 'Bangalore Urban', 'Karnataka', 'https://electrotech.com', NULL, 4, 1, 1, 1, 1, '2021-04-01', 'Active', 'Private', 'SubCat1', 'ClassA', 'Listed', 'ROC-Bangalore', '2710', 1, 1, GETDATE(), 1, '127.0.0.1');

-- Populate Company Products
INSERT INTO CompanyProducts (CompanyMaster_FK_Id, CompanyRefNo, ProductRefNo, ProductName, ProductDesc, ProductType_Fk_Id, ItemExported, Final_Submit, IsActive, CreatedBy, CreatedDate, CreatedIP)
VALUES
(1, 'CMP001', 'PRD001', 'High Voltage Transformer', 'Substation voltage regulation', 1, 1, 1, 1, 10, GETDATE(), '127.0.0.1');
