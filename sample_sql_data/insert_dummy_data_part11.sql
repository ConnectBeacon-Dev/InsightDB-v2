
-- Populating Core Expertise Master
INSERT INTO CompanyCoreExpertiseMaster (CoreExpertiseName, CreatedBy, CreatedDate, IsActive) VALUES
('Electrical Engineering', 'admin', GETDATE(), 1),
('Mechanical Engineering', 'admin', GETDATE(), 1),
('Aerospace', 'admin', GETDATE(), 1),
('Telecom', 'admin', GETDATE(), 1),
('Software Development', 'admin', GETDATE(), 1);

-- Populating Industry Domain Master
INSERT INTO IndustryDomainMaster (IndustryDomainType, CreatedDate, CreatedBy, IsActive) VALUES
('Energy', GETDATE(), 'admin', 1),
('Manufacturing', GETDATE(), 'admin', 1),
('Aerospace', GETDATE(), 'admin', 1),
('Telecommunication', GETDATE(), 'admin', 1),
('IT Services', GETDATE(), 'admin', 1);

-- Populating Industry Subdomain Master
INSERT INTO IndustrySubdomainType (IndustrySubDomainName, CreatedDate, CreatedBy, IsActive) VALUES
('Power Distribution', GETDATE(), 'admin', 1),
('Automobile Parts', GETDATE(), 'admin', 1),
('Satellites', GETDATE(), 'admin', 1),
('5G Infrastructure', GETDATE(), 'admin', 1),
('Cloud Computing', GETDATE(), 'admin', 1);

-- Populating Organisation Types
INSERT INTO OrganisationTypeMaster (Organization_Type, CreatedDate, CreatedBy, IsActive) VALUES
('Private Ltd', GETDATE(), 'admin', 1),
('Public Ltd', GETDATE(), 'admin', 1),
('LLP', GETDATE(), 'admin', 1),
('Government', GETDATE(), 'admin', 1),
('NGO', GETDATE(), 'admin', 1);

-- Populating Scale Master
INSERT INTO ScaleMaster (CompanyScale, CreatedDate, CreatedBy, IsActive) VALUES
('Micro', GETDATE(), 'admin', 1),
('Small', GETDATE(), 'admin', 1),
('Medium', GETDATE(), 'admin', 1),
('Large', GETDATE(), 'admin', 1),
('Enterprise', GETDATE(), 'admin', 1);

-- Populating Country Master
INSERT INTO CountryMaster (CountryName, DisplayCountryName, IsActive, CreatedBy, CreatedDate, CreatedByIP, UpdatedBy, UpdatedDate, UpdatedByIP) VALUES
('India', 'India', 1, 1, GETDATE(), '127.0.0.1', 1, GETDATE(), '127.0.0.1'),
('Sweden', 'Sweden', 1, 1, GETDATE(), '127.0.0.1', 1, GETDATE(), '127.0.0.1'),
('Japan', 'Japan', 1, 1, GETDATE(), '127.0.0.1', 1, GETDATE(), '127.0.0.1'),
('China', 'China', 1, 1, GETDATE(), '127.0.0.1', 1, GETDATE(), '127.0.0.1');
