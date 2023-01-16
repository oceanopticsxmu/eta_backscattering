%% Synthetic data
% references: IOCCG(2006)
 
clc; 
clear all;

N=400000;
%% read aph from climotology aph product (MODIS + GIOP)
aph_cl=ncread('A20021852021031.L3m_CU_IOP_aph_443_giop_4km.nc','aph_443_giop');
aph_cl(aph_cl<0)=NaN;
aph_cl=aph_cl(:);
idx=find(isnan(aph_cl(:))==0);

A=randperm(length(idx));
B=sort(A(1:N));
aph440=aph_cl(idx(B));


% N=50;
% aph440=aph_cl(idx(1:N:end));

% rdix=randperm(length(idx));
% aph1=sort(aph440(1:10000));
close all;
histogram(aph440);
set(gca,'xscale','log','fontname','Times new Roman','fontsize',13)     
xlim([1e-3,1e1]);
ylim tight;
yl = ylim;

txt=['Max=' num2str(round(max(aph440),2)) ' m^{-1} (N=' num2str(N) ')'];
text(5e-2,0.8*yl(2),txt,'fontname','Times new Roman','fontsize',14);
% title(char(ti(i)),'fontname','Times new Roman','fontsize',14);
minap=min(aph440)
maxap=max(aph440)

C = (20.*aph440).^(1/0.626);
% max(C)
% min(C)
aph = [];


%% read group aph
filename = 'aphclasses350-800.xlsx';
wavelength = 350:5:800;
for i = 1 : 12
    [num,~] = xlsread(filename, i);
    aph = num(2:end, 2:end);
    aph_plus{i} = aph./aph(:,wavelength == 440);
end

%% set group range by aph440
g_range = [0, 0.005;
    0.005, 0.01;
    0.01, 0.015;
    0.015, 0.02;
    0.02, 0.03;
    0.03, 0.05;
    0.05, 0.07;
    0.07, 0.12;
    0.12, 0.2;
    0.2, 0.3;
    0.3, 1.5;
    1.5, 40];

%% create aph by group aph_plus
	% first, set  aph440 in a range of 0.001 to 10 m-1; 
    % then divide this range into 10000 scales equally in Log space. 
    % The rest follow the descriptions in the document

% n_create_samples = 200000; % create 100 pairs of data. 
% minap=0.002;
% maxap=1; 
% aph440_1 = linspace(minap, maxap, n_create_samples)';
% aph440_2 = logspace(log10(minap), log10(maxap), n_create_samples)';
% aph440=[aph440_1;aph440_2];
% aph440 = logspace(log10(minap), log10(maxap), n_create_samples)';
% aph440  = linspace(minap, maxap, n_create_samples)';



for i = 1: size(g_range, 1)
    g_aph440 = aph440(aph440 > g_range(i,1) & aph440 <= g_range(i,2));
    tem_aph_plus = aph_plus{i};
    n_aph = size(tem_aph_plus, 1);
    tem_rows = ceil(rand(length(g_aph440), 1) * n_aph);
    aph_i = tem_aph_plus(tem_rows, :) .* g_aph440;
    aph = [aph; aph_i];
end
% interpolation
wavelength_interp = 400:2:800; 
aph_interp = zeros(size(aph, 1), length(wavelength_interp));
for i = 1 : size(aph, 1)
    aph_interp(i, :) = interp1(wavelength, aph(i, :), wavelength_interp, 'pchip');
end
aph = aph_interp;
wavelength = wavelength_interp;

%% synthetic Rrs
[Rrs, a, adm, ag, bb, bbp, bbdm, bbph, Kd, Y, C] = aph2Rrs2(aph, wavelength);
max(Y)
min(Y)

wlnew=[400 412 443 488 531 547 667 678];
for iwl=1:length(wlnew)
    [~,wlidx(iwl)]=min(abs(wavelength-wlnew(iwl)));
end



Syn.wl=wlnew;
Syn.Rrs=round(Rrs(:,wlidx),6);
Syn.a=round(a(:,wlidx),5);
Syn.bbp=round(bbp(:,wlidx),6);
% Syn.Kd=round(Kd(:,wlidx),4);
Syn.Y=round(Y,3);
Syn.C=round(C,3);

Syn.info= ['Y in a range of' num2str(min(round(Y,3))) '-' num2str(max(round(Y,3))) '; aph in a range of  ' num2str(minap) '-' num2str(maxap) ' climo_aph'];

save('SynDataST.mat','Syn');


bbp443=bbp(:,wlidx(3));
figure;

subplot(1,3,1)
histogram(aph440);
set(gca,'xscale','log');
ylim tight 
title('aph440 hist');
subplot(1,3,2)
histogram(log(bbp443));
set(gca,'xscale','log');
xlim([1e-4 1e0])
ylim tight 
title('{\itb_{bp}}(443) hist');
subplot(1,3,3)
histogram(Y,'Normalization','pdf');
xlabel('Simulated \it\eta','fontname','Times New Roman','fontsize',13);
ylabel('Density','fontname','Times New Roman','fontsize',13);
title('{\it\eta} histogram','fontname','Times New Roman','fontsize',15);
set(gca,'fontname','Times New Roman','fontsize',13);




xlsfile='eta_vs_chla_bbp.xlsx';  
[data,~,alldata]=xlsread(xlsfile);
x1=data(:,3);
y1=data(:,2);
idx1=data(:,1);

x2=data(:,5);
y2=data(:,6);
idx2=data(:,4);


figure;
set(gcf,'Position',[100 50 700 320]);

ti={'a){\it\eta} v.s. Chla';'b) {\it\eta} v.s. {\itb_{bp}}(443)'};
mc=[0.00,0.45,0.74;1.00,0.41,0.16;0,0,0;0,1,0;0.00,0.45,0.74;1.00,0.41,0.16];
mk={'o';'s';'+';'*';'^';'>'}; 

wwl=Syn.wl;
YY=Syn.Y;
BB=Syn.bbp;
[~,idx]=min(abs(wwl-443));
b443=BB(:,idx);

plot(b443,YY,'.','color',[.8,.8,.8]);
hold on;

for i=1:6
    idx=find(idx2==i);
    x_temp=x2(idx);
    y_temp=y2(idx);  
    scatter(x_temp,y_temp,30,char(mk(i)),'MarkerFacecolor',mc(i,:),'MarkerEdgecolor',mc(i,:),'MarkerFaceAlpha',0.8);     
    box on;
    hold on;    
end
  ylim([-0.5 3]);
  xlim([1e-4,1e-1]);
  ytick=[-0.5,0,1,2,3];
  xtick=[1e-4,1e-3,1e-2,1e-1];
  set(gca,'xscale','log','ytick',ytick,'yticklabel',ytick,'xtick',xtick,'xticklabel',xtick,'fontsize',13,'fontname','Times New Roman','linewidth',1,'XMinorTick','on','YMinorTick','on','xMinorGrid','on');
  xlabel('{\itb_{bp}}(443) (m^{-1})','fontsize',14,'fontname','Times New Roman');
  ylabel('{\it\eta}','fontsize',14,'fontname','Times New Roman','fontweight','bold');
 



