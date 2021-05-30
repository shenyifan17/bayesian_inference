N=400; % number of data points - you can increase this if you want to
       % learn better features (but it will take longer).
D=16; % dimensionality of the data

rand('state',0);

% Define the basic shapes of the features

m1=[0 0 1 0;
    0 1 1 1;
    0 0 1 0;
    0 0 0 0]; 

m2=[0 1 0 0;
    0 1 0 0;
    0 1 0 0;
    0 1 0 0];

m3=[1 1 1 1;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0];

m4=[1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1]; 

m5=[0 0 0 0;
    0 0 0 0;
    1 1 0 0;
    1 1 0 0]; 

m6=[1 1 1 1;
    1 0 0 1;
    1 0 0 1;
    1 1 1 1]; 

m7=[0 0 0 0;
    0 1 1 0;
    0 1 1 0;
    0 0 0 0];

m8=[0 0 0 1;
    0 0 0 1;
    0 0 0 1;
    0 0 0 1];

nfeat=8; % number of features
rr=0.5+rand(nfeat,1)*0.5; % weight of each feature between 0.5 and 1
mut=[rr(1)*m1(:) rr(2)*m2(:) rr(3)*m3(:) rr(4)*m4(:) rr(5)*m5(:) ...
rr(6)*m6(:) rr(7)*m7(:) rr(8)*m8(:)]';
s=rand(N,nfeat)<0.3; % each feature occurs with prob 0.3 independently 

% Generate Data - The Data is stored in Y

Y=s*mut+randn(N,D)*0.1; % some Gaussian noise is added 

% Plot a 13x13 matrix of images 
% set(gcf,'Color',[0.2 0.4 0.6]); % Background color
% colormap gray;
% k=0;
% nrows=13;
% for i=1:nrows
%   for j=1:nrows
%     k=k+1;
%     subplot(nrows,nrows,k);
%     imagesc(reshape(Y(k,:),4,4),[0 2]);
%     axis off;
%     axis equal;
%   end
% end





