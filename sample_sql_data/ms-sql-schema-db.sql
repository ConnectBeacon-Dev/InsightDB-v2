DROP DATABASE TestDB;
GO

CREATE DATABASE TestDB;
GO

USE TestDB;
GO

CREATE TABLE CompanyCoreExpertiseMaster (Id INT PRIMARY KEY IDENTITY(1,1), CoreExpertiseName NVARCHAR(50), CreatedBy NVARCHAR(150), CreatedDate DATETIME, IsActive BIT);
GO
SELECT * FROM CompanyCoreExpertiseMaster;
GO

CREATE TABLE IndustrySubdomainType (Id INT PRIMARY KEY IDENTITY(1,1), IndustrySubDomainName NVARCHAR(200), CreatedDate DATETIME, CreatedBy NVARCHAR(150), IsActive BIT);
GO
SELECT * FROM IndustrySubdomainType;
GO

CREATE TABLE IndustryDomainMaster (Id INT PRIMARY KEY IDENTITY(1,1), IndustryDomainType NVARCHAR(200), CreatedDate DATETIME, CreatedBy NVARCHAR(150), IsActive BIT);
GO
SELECT * FROM IndustryDomainMaster;
GO

CREATE TABLE OrganisationTypeMaster (Id INT PRIMARY KEY IDENTITY(1,1), Organization_Type NVARCHAR(250), CreatedDate DATETIME, CreatedBy NVARCHAR(150), IsActive BIT);
GO
SELECT * FROM OrganisationTypeMaster;
GO

CREATE TABLE ScaleMaster (Id INT PRIMARY KEY IDENTITY(1,1), CompanyScale NVARCHAR(50), CreatedDate DATETIME, CreatedBy NVARCHAR(150), IsActive BIT);
GO
SELECT * FROM ScaleMaster;
GO

CREATE TABLE CountryMaster (Id BIGINT PRIMARY KEY IDENTITY(1,1), CountryName NVARCHAR(170), DisplayCountryName NVARCHAR(150), IsActive BIT, CreatedBy INT, CreatedDate DATETIME, CreatedByIP NVARCHAR(50), UpdatedBy SMALLINT, UpdatedDate DATETIME, UpdatedByIP NVARCHAR(50));
GO
SELECT * FROM CountryMaster;
GO

-- CompanyMaster is large; no line wrapping here to preserve SQLcmd compatibility.
CREATE TABLE CompanyMaster (Id INT PRIMARY KEY IDENTITY(1,1), CINNumber VARCHAR(100), Pan NVARCHAR(500), CompanyRefNo NVARCHAR(100), DPSU_Fk_Id NVARCHAR(50), CompanyName NVARCHAR(500), POC_Email NVARCHAR(500), Phone NVARCHAR(500), EmailId NVARCHAR(500), Address VARCHAR(MAX), CityName NVARCHAR(500), PinCode NVARCHAR(500), Country_Fk_id BIGINT, District NVARCHAR(500), State NVARCHAR(500), Website NVARCHAR(500), Logo NVARCHAR(500), CompanyScale_Fk_Id INT, CompanyType_Fk_Id INT, IndustryDomain_Fk_Id INT, IndustrySubDomain_Fk_Id INT, CompanyCoreExpertise_Fk_Id INT, CompanyRegistrationDate VARCHAR(100), CompanyStatus NVARCHAR(100), CompanyCategory NVARCHAR(100), CompanySubCategory NVARCHAR(100), CompanyClass NVARCHAR(100), ListingStatus NVARCHAR(50), CompanyROC NVARCHAR(100), CompanyIndustrialClassification NVARCHAR(200), OtherScale NVARCHAR(100), OtherCompanyType NVARCHAR(100), OtherCompanyCoreExpertise NVARCHAR(50), OtherCompIndDomain NVARCHAR(50), OtherCompIndSubDomain NVARCHAR(50), Final_Submit INT, IsActive BIT, CreatedDate DATETIME, CreatedBy INT, IPAddress VARCHAR(50), CONSTRAINT fk_country FOREIGN KEY (Country_Fk_id) REFERENCES CountryMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_company_scale FOREIGN KEY (CompanyScale_Fk_Id) REFERENCES ScaleMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_org_type FOREIGN KEY (CompanyType_Fk_Id) REFERENCES OrganisationTypeMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_industry_domain FOREIGN KEY (IndustryDomain_Fk_Id) REFERENCES IndustryDomainMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_industry_subdomain FOREIGN KEY (IndustrySubDomain_Fk_Id) REFERENCES IndustrySubdomainType(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_core_expertise FOREIGN KEY (CompanyCoreExpertise_Fk_Id) REFERENCES CompanyCoreExpertiseMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL);
GO
SELECT * FROM CompanyMaster;
GO

CREATE TABLE RDCategoryMaster (Id INT PRIMARY KEY IDENTITY(1,1), RDCategoryName NVARCHAR(50), IsActive BIT);
GO
SELECT * FROM RDCategoryMaster;
GO

CREATE TABLE RDSubCategoryMaster (Id INT PRIMARY KEY IDENTITY(1,1), RDCategory_Fk_Id INT, RDSubCategoryName NVARCHAR(500), IsActive BIT, CONSTRAINT fk_rd_subcat_category FOREIGN KEY (RDCategory_Fk_Id) REFERENCES RDCategoryMaster(Id) ON UPDATE CASCADE ON DELETE NO ACTION);
GO
SELECT * FROM RDSubCategoryMaster;
GO

CREATE TABLE CompanyRDFacility (Id INT PRIMARY KEY IDENTITY(1,1), CompanyMaster_FK_ID INT, CompanyRefNo NVARCHAR(200), RDRefNo NVARCHAR(200), RDCategory_Fk_ID INT, RDSubCategory_Fk_Id INT, RD_Details NVARCHAR(MAX), IsNabIAccredited BIT, IsActive BIT, Final_Submit BIT, CreatedBy NVARCHAR(150), CreatedDate DATETIME, CreatedIP NVARCHAR(250), UpdatedBy NVARCHAR(150), UpdateDate DATETIME, UpdatedIP NVARCHAR(250), CONSTRAINT fk_company_master FOREIGN KEY (CompanyMaster_FK_ID) REFERENCES CompanyMaster(Id) ON UPDATE CASCADE ON DELETE NO ACTION, CONSTRAINT fk_rd_category FOREIGN KEY (RDCategory_Fk_ID) REFERENCES RDCategoryMaster(Id) ON UPDATE CASCADE ON DELETE NO ACTION, CONSTRAINT fk_rd_subcategory FOREIGN KEY (RDSubCategory_Fk_Id) REFERENCES RDSubCategoryMaster(Id) ON UPDATE NO ACTION ON DELETE NO ACTION);
GO
SELECT * FROM CompanyRDFacility;
GO

CREATE TABLE YearMaster (Id INT PRIMARY KEY IDENTITY(1,1), Year NVARCHAR(20), IsActive BIT);
GO
SELECT * FROM YearMaster;
GO

CREATE TABLE CertificationTypeMaster (Id BIGINT PRIMARY KEY IDENTITY(1,1), Cert_Type NVARCHAR(51), IsActive BIT);
GO
SELECT * FROM CertificationTypeMaster;
GO

CREATE TABLE CompanyCertificationDetail (Id INT PRIMARY KEY IDENTITY(1,1), CompanyMaster_Fk_Id INT, CompanyRefNo NVARCHAR(200), CertificateRefNo NVARCHAR(200), Certification_Type NVARCHAR(200), OtherCertification_Type INT, Certificate_No NVARCHAR(200), Certificate_StartDate DATE, Certificate_EndDate DATE, Final_Submit BIT, IsActive BIT, CreatedBy NVARCHAR(50), CreatedDate DATETIME, CreatedIP VARCHAR(250), UpdatedBy NVARCHAR(150), UpdatedDate DATETIME, UpdatedIP NVARCHAR(250), CertificateType_Fk_Id BIGINT, CONSTRAINT fk_cert_company FOREIGN KEY (CompanyMaster_Fk_Id) REFERENCES CompanyMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_cert_type FOREIGN KEY (CertificateType_Fk_Id) REFERENCES CertificationTypeMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL);
GO
SELECT * FROM CompanyCertificationDetail;
GO

CREATE TABLE TestFacilityCategoryMaster (Id INT PRIMARY KEY IDENTITY(1,1), CategoryName NVARCHAR(200), IsActive BIT);
GO
SELECT * FROM TestFacilityCategoryMaster;
GO

CREATE TABLE TestFacilitySubCategoryMaster (Id INT PRIMARY KEY IDENTITY(1,1), TestFacility_Fk_Id INT, SubCategoryName NVARCHAR(500), Description NVARCHAR(100), IsActive BIT, CONSTRAINT fk_testfacility_subcat_category FOREIGN KEY (TestFacility_Fk_Id) REFERENCES TestFacilityCategoryMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL);
GO
SELECT * FROM TestFacilitySubCategoryMaster;
GO

CREATE TABLE CompanyTestFacility (Id INT PRIMARY KEY IDENTITY(1,1), CompanyMaster_FK_ID INT, CompanyRefNo NVARCHAR(200), TestRefNo NVARCHAR(200), TestDetails NVARCHAR(MAX), Final_Submit BIT, IsActive BIT, CreatedBy VARCHAR(150), CreatedDate DATETIME, CreatedIP NVARCHAR(50), UpdatedBy NVARCHAR(150), UpdatedDate DATETIME, UpdateIP NVARCHAR(50), IsNabIAccredited BIT, TestFacilityCategory_Fk_Id INT, TestFacilitySubCategory_Fk_id INT, CONSTRAINT fk_company_test_facility FOREIGN KEY (CompanyMaster_FK_ID) REFERENCES CompanyMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_test_facility_cat FOREIGN KEY (TestFacilityCategory_Fk_Id) REFERENCES TestFacilityCategoryMaster(Id), CONSTRAINT fk_testfacility_subcat FOREIGN KEY (TestFacilitySubCategory_Fk_id) REFERENCES TestFacilitySubCategoryMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL);
GO
SELECT * FROM CompanyTestFacility;
GO

CREATE TABLE ProductTypeMaster (Id INT PRIMARY KEY IDENTITY(1,1), ProductTypeName VARCHAR(1000), IsActive BIT);
GO
SELECT * FROM ProductTypeMaster;
GO

CREATE TABLE PlatformTechAreaMaster (Id INT PRIMARY KEY IDENTITY(1,1), PTAName NVARCHAR(500), IsActive BIT);
GO
SELECT * FROM PlatformTechAreaMaster;
GO

CREATE TABLE DefencePlatformMaster (Id INT PRIMARY KEY IDENTITY(1,1), Name_of_Defence_Platform NVARCHAR(60), IsActive BIT);
GO
SELECT * FROM DefencePlatformMaster;
GO

CREATE TABLE CompanyProducts (Id BIGINT PRIMARY KEY IDENTITY(1,1), CompanyMaster_FK_Id INT, CompanyRefNo NVARCHAR(50), ProductRefNo NVARCHAR(70), ProductName NVARCHAR(500), ProductDesc NVARCHAR(1500), NSNNumber NVARCHAR(50), HSNCode NVARCHAR(500), FutureExpansion NVARCHAR(100), SIUnit_Fk_Id INT, AnnualProductionCapacity NVARCHAR(100), ProductImage NVARCHAR(500), ProductCertificateDet NVARCHAR(1000), ProductType_Fk_Id INT, DefencePlatform_Fk_Id INT, PTAType_Fk_Id INT, SalientFeature NVARCHAR(500), ItemExported BIT, Final_Submit BIT, IsActive BIT, CreatedBy BIGINT, CreatedDate DATETIME, CreatedIP VARCHAR(50), UpdatedBy BIGINT, UpdatedDate DATETIME, UpdatedIP VARCHAR(50), CONSTRAINT fk_product_company FOREIGN KEY (CompanyMaster_FK_Id) REFERENCES CompanyMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_product_type FOREIGN KEY (ProductType_Fk_Id) REFERENCES ProductTypeMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_defence_platform FOREIGN KEY (DefencePlatform_Fk_Id) REFERENCES DefencePlatformMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL, CONSTRAINT fk_pta_type FOREIGN KEY (PTAType_Fk_Id) REFERENCES PlatformTechAreaMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL);
GO
SELECT * FROM CompanyProducts;
GO

CREATE TABLE CompanyTurnOver (Id INT PRIMARY KEY IDENTITY(1,1), Company_FK_Id INT, YearId NVARCHAR(1000), Amount NVARCHAR(2000), Status NVARCHAR(250), Final_Submit BIT, CreatedBy NVARCHAR(150), CreatedDate DATETIME, CreatedIP NVARCHAR(250), UpdatedBy NVARCHAR(150), UpdatedDate DATETIME, UpdateIP NVARCHAR(250), IsActive BIT, CONSTRAINT fk_turnover_company FOREIGN KEY (Company_FK_Id) REFERENCES CompanyMaster(Id) ON UPDATE CASCADE ON DELETE SET NULL);
GO
SELECT * FROM CompanyTurnOver;
GO
