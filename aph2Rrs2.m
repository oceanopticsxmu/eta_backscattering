function [Rrs, a, adm, ag, bb, bbp, bbdm, bbph, Kd, Y, C] = aph2Rrs2(aph, wavelength)
%% notes
% generate IOPs and AOPs from known spectral phytoplankton coefficient 
% reference: IOCCG-OCAG (2003), Model, parameters, and approaches that used to generate wide range of absorption and backscattering spectraRep., 
% International Ocean Colour Coordinating Group, http://www.ioccg.org/groups/OCAG_data.html.

% Inputs: spectral phytoplankton absorption coefficients aph(lambda) & corresponding wavelengths

% outputs: Remote sensing reflectance: Rrs [sr-1]
%          absorption coefficients a (total & components) : a, adm,ag
%          backscattering  coefficients: bb bbdm  bbph bbp 
%          diffuse attanuation coefficient Kd(lambda).
%          spectral slope of bbp --> Y

%          ph,g,dm,p stand for different water constituents of phytoplankton, colored disolved organic
%          matter CDOM, detritus and minerals, and particles, respectively.

% drafted by Wendian Lai, ~2021
% modified by Xiaolong Yu, 2022-02-16

%% generate 
%%
[sno, wlno] = size(aph);  % sample number and wavelength number 
aph440 = aph(:, wavelength == 440);
C = (20.*aph440).^(1/0.626);

%% a
%% adm
Sdm = 0.007 + (0.015 - 0.007) .* rand(sno, 1);   % Sdm radomly between 0.007 and 0.015 [Babin et al., 2003; Roesler etal., 1989]
p1 = 0.1 + (0.5 .* rand(sno, 1) .* aph440) ./ (0.05 + aph440);
adm440 = p1 .* aph440;
repSdm = repmat(Sdm, 1, wlno);
repadm440 = repmat(adm440, 1, wlno);
repwavelength = repmat(wavelength, sno, 1);
adm = repadm440 .* exp(-repSdm .* (repwavelength - 440));

%% ag
Sg = 0.01 + (0.03 - 0.01) .* rand(sno,1); % Sg randomly  between 0.01 – 0.02 nm-1 [Babin et al., 2003; Kirk, 1994]
p2 = 0.3 + (5.7.*rand(sno,1).* aph440)./(0.02+aph440);  % 0.3 - 6 

ag440 = p2 .* aph440; 
repag440 = repmat(ag440, 1, wlno);
repwavelength = repmat(wavelength, sno, 1);
repSg = repmat(Sg, 1, wlno);
ag = repag440 .* exp(-repSg .* (repwavelength - 440));

%% bb
%% bbph
% n1 = -0.4 + (2.4 + 1.4 .* rand(sno, 1)) ./ (1 + C .^(0.5)); % n1 is in a range of -0.1 to 2.2 
% n1 = -0.7 + (3.0 + 1.4 .* rand(sno, 1)) ./ (1 + C .^(0.5)); % n1 is in a range of -0.1 to 2.2 
n1 = -0.7 + (3.0 + 1.5 .* rand(sno, 1)) ./ (1 + C .^(0.5)); % n1 is in a range of -0.1 to 2.2 used in ms

p3 = 0.06 + (0.6 - 0.06) .* rand(sno, 1);
cph550 = p3 .* C.^0.57;
% cal bbph
repn1 = repmat(n1, 1, wlno);
repcph550 = repmat(cph550, 1, wlno);
repwavelength = repmat(wavelength, sno, 1);
cph = repcph550 .* (550 ./ repwavelength) .^ repn1;
bbph = 0.01 .* (cph - aph);

%% bbdm
% n2 = -0.5 + (2.2 + 1.4.* rand(sno, 1)) ./ (1 + C.^0.5); % n2 in a range of -0.5 - 2.2. the original is 2.0
% n2 = -0.6 + (2.2 + 1.5.* rand(sno, 1)) ./ (1 + C.^0.5); % n2 in a range of -0.5 - 2.2. the original is 2.0
n2 = -0.6 + (2.2 + 1.5.* rand(sno, 1)) ./ (1 + C.^0.5); % n2 in a range of -0.5 - 2.2. the original is 2.0 used in ms

p4 = 0.06 + (0.6 - 0.06) .* rand(sno, 1);
bdm550 = p4 .* C .^ 0.766;
% cal bbdm
repbdm550 = repmat(bdm550, 1, wlno);
repwavelength = repmat(wavelength,sno, 1);
repn2 = repmat(n2, 1, wlno);
bbdm = 0.0183 .* repbdm550 .* (550 ./ repwavelength) .^ repn2;

%% aw & bbw
aw=h2o_iops_Zhh_lee(wavelength,'a');
[~,~,bsw]= betasw_zhh2009(wavelength,15,90,35,0.039);
bbw=0.5*bsw;
% bbw=h2o_iops_Zhh_lee(wavelength,'bb');
repaw = repmat(aw,sno,1);
repbbw = repmat(bbw,sno,1);

%% a & bb
a = repaw + aph + adm + ag;
anw = aph + adm + ag;
bb = repbbw + bbdm + bbph;
% ap = aph + adm;
bbp = bbdm + bbph;

bnw=bbdm./0.0183+bbph./0.01 ;

cnw=anw+bnw;
%% cal Rrs
gw = 0.113; 
G0 = 0.197;
G1 = 0.636;
G2 = 2.552;
gp = G0.* (1 - G1 .* exp(-G2 .* bbp ./ (a + bb)));
rrs = gw .* bbw ./ (a + bb) + gp .* bbp ./ (a + bb);
Rrs = 0.52 .* rrs ./ (1 - 1.7 .* rrs);

% QAA_forward
% g0 = 0.0895; 
% g1 = 0.1247;
% u = bb ./ (a + bb);
% rrs = ((2.*g1.*u + g0) .^ 2 - g0.^2) / (4 .* g1);
% Rrs2 = (0.5 .* rrs) ./ (1 - 1.7.*rrs);

%% cal Kd
gama = 0.265;
m0 = 0.005;
m1 = 4.259; m2 = 0.52; m3 = 10.8;
SZA = 30;
Kd = (1+m0.*SZA).*a + (1-gama.*(bbw./bb)).*m1 .* (1-m2.*exp(-m3.*a)).*bb;


%% a555 & Y  
a550 = a(:,wavelength==550);
repbbp550=repmat(bbp(:,wavelength == 550),1,wlno); 
Y = log(bbp./repbbp550)./log(550./repwavelength);
Y = nanmean(Y, 2);  % Statistically, this Y is closer to the one using optimization Y1
% 
% Y = cal_Y_fit(bbp, wavelength);  % optimization
% [Y,~] = cal_Y_log_Model2(bbp, wavelength); % linear regression of log-transformed data

end

