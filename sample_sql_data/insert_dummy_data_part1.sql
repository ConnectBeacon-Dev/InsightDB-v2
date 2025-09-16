INSERT INTO CompanyCoreExpertiseMaster (CoreExpertiseName, CreatedBy, CreatedDate, IsActive) VALUES
(N'Electrical Engineering', 'admin', GETDATE(), 1),
(N'Mechanical Engineering', 'admin', GETDATE(), 1),
(N'Aerospace', 'admin', GETDATE(), 1),
(N'Telecom', 'admin', GETDATE(), 1),
(N'Software Development', 'admin', GETDATE(), 1);

INSERT INTO IndustryDomainMaster (IndustryDomainType, CreatedDate, CreatedBy, IsActive) VALUES
(N'Energy', GETDATE(), 'admin', 1),
(N'Manufacturing', GETDATE(), 'admin', 1),
(N'Aerospace', GETDATE(), 'admin', 1),
(N'Telecommunication', GETDATE(), 'admin', 1),
(N'IT Services', GETDATE(), 'admin', 1);

INSERT INTO IndustrySubdomainType (IndustrySubDomainName, CreatedDate, CreatedBy, IsActive) VALUES
(N'Power Distribution', GETDATE(), 'admin', 1),
(N'Automobile Parts', GETDATE(), 'admin', 1),
(N'Satellites', GETDATE(), 'admin', 1),
(N'5G Infrastructure', GETDATE(), 'admin', 1),
(N'Cloud Computing', GETDATE(), 'admin', 1);

INSERT INTO OrganisationTypeMaster (Organization_Type, CreatedDate, CreatedBy, IsActive) VALUES
(N'Private Ltd', GETDATE(), 'admin', 1),
(N'Public Ltd', GETDATE(), 'admin', 1),
(N'LLP', GETDATE(), 'admin', 1),
(N'Government', GETDATE(), 'admin', 1),
(N'NGO', GETDATE(), 'admin', 1);

INSERT INTO ScaleMaster (CompanyScale, CreatedDate, CreatedBy, IsActive) VALUES
(N'Micro', GETDATE(), 'admin', 1),
(N'Small', GETDATE(), 'admin', 1),
(N'Medium', GETDATE(), 'admin', 1),
(N'Large', GETDATE(), 'admin', 1),
(N'Enterprise', GETDATE(), 'admin', 1);

INSERT INTO CountryMaster (CountryName, DisplayCountryName, IsActive, CreatedBy, CreatedDate, CreatedByIP, UpdatedBy, UpdatedDate, UpdatedByIP) VALUES
(N'India', N'India', 1, 1, GETDATE(), '127.0.0.1', 1, GETDATE(), '127.0.0.1'),
(N'Sweden', N'Sweden', 1, 1, GETDATE(), '127.0.0.1', 1, GETDATE(), '127.0.0.1'),
(N'Japan', N'Japan', 1, 1, GETDATE(), '127.0.0.1', 1, GETDATE(), '127.0.0.1'),
(N'China', N'China', 1, 1, GETDATE(), '127.0.0.1', 1, GETDATE(), '127.0.0.1');

