function [net, info] = cnn_cifar_riem2(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet_nl' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnn, 'data', ...
  sprintf('cifar-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data','cifar') ;
opts.evalMode = 1;
if opts.evalMode == 1
    opts.imdbPath = fullfile(opts.expDir,'imdbEval.mat');
else
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
end 
opts.special = 1;
opts.first_label = 4;
opts.second_label = 6;

opts.whitenData = false ;
opts.contrastNormalization = false ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'lenet'
    net = cnn_cifar_init('networkType', opts.networkType) ;
  case 'lenet_nl'
    net = cnn_cifar_init_nl('networkType', opts.networkType) ;
  case 'lenet_nl_temp'
    net = cnn_cifar_init_nl('networkType', opts.networkType) ;    
  case 'lenet_nl_bn'
    net = cnn_cifar_init_nl_bn('networkType', opts.networkType) ;
  case 'nin'
    net = cnn_cifar_init_nin('networkType', opts.networkType) ;
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
  imdb.images.data = single(imdb.images.data);
else
    if ~opts.evalMode
        imdb = getCifarImdb(opts) ;
    else
        imdb = getCifarImdbEval(opts);
    end 
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

%for ix=1:1000
%figure(1);imshow(imdb.images.data(:,:,:,ix))
%pause(0.1)
%end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3),...
  'imdbEval', opts.evalMode) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));
labels = single(cat(2, labels{:}));

if opts.special == 1
	valid_label_idx = find((labels == (opts.first_label)) | (labels == (opts.second_label)));
	data = data(:,:,:,valid_label_idx);
	set = set(valid_label_idx);
	valid_labels = labels(valid_label_idx);
	valid_labels(find(valid_labels == (opts.first_label))) = 1;
    valid_labels(find(valid_labels == (opts.second_label))) = 2;
    labels = valid_labels;
end 

numlabels=numel(labels);

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],numlabels) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],numlabels) ;
  W = z(:,set == 1)*z(:,set == 1)'/numlabels ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(labels) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;



% -------------------------------------------------------------------------
function imdb = getCifarImdbJitter(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));
labels = single(cat(2, labels{:}));

if opts.special == 1
	valid_label_idx = find((labels == (opts.first_label)) | (labels == (opts.second_label)));
	data = data(:,:,:,valid_label_idx);
	set = set(valid_label_idx);
	valid_labels = labels(valid_label_idx);
	valid_labels(find(valid_labels == (opts.first_label))) = 1;
    valid_labels(find(valid_labels == (opts.second_label))) = 2;
    labels = valid_labels;
end 

outIdx = 1;
dataOut = [];
for dataIdx=1:size(data,4)
    for numJitter=1:4
        tx = randn * 15;
        ty = randn * 15;
        dataOut(:,:,:,outIdx) = imtranslate(data(:,:,:,dataIdx),[tx, ty],'FillValues',255);
        outIdx = outIdx + 1;
    end 
end 

data = dataOut;

setX = repmat(set(:),1,4);
setY = setX';
set = setY(:);
set = set';

labelsX = repmat(labels(:),1,4);
labelsY = labelsX';
labels = labelsY(:);
labels = labels';

numlabels=numel(labels);

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],numlabels) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],numlabels) ;
  W = z(:,set == 1)*z(:,set == 1)'/numlabels ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(labels) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;

% -------------------------------------------------------------------------
function imdb = getCifarImdbEval(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));

for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);

v = VideoReader('C:\Users\piyushk\Desktop\morph\cat_dog\cat_dog_backforth_smooth_300.avi');
currAxes = axes;
v.CurrentTime = 0;
itx = 1;
max_tx = 15;
min_tx = -15;
steps_per_tx = 0.1;%
tx_range = min_tx:steps_per_tx:max_tx;
tx_used = zeros(1,numel(tx_range));
ty = 0;
data_vid = [];
total = 1;
while hasFrame(v)
    vidFrame = readFrame(v);
    %image(vidFrame, 'Parent', currAxes);
    %currAxes.Visible = 'off';
    %pause(1/v.FrameRate);        
    outimg = imresize(vidFrame,[32, 32]);
    for tx=tx_range
        outimg2 = imtranslate(outimg,[tx, ty],'FillValues',255);
        data_vid(:,:,:,total ) = outimg2;
        tx_used(total ) = tx;
        theta_used(total ) = itx;        
        %fprintf('.')
        total = total + 1;
    end 
    itx = itx + 1;      
    fprintf('\n %d : %d', itx, total);
end

data = bsxfun(@minus, single(data_vid), dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng
num_images = size(data,4);
set = 3*ones(1,num_images);
set(end) = 1;
labels = ones(1,num_images);

if opts.contrastNormalization
  z = reshape(data,[],num_images) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],num_images) ;
  W = z(:,set == 1)*z(:,set == 1)'/num_images ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = labels;%single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
imdb.images.tx_used = tx_used;
imdb.images.theta_used = theta_used;


% -------------------------------------------------------------------------
function imdb = getCifarImdbEval2(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));

for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));
data_orig = data;

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
cat1 = imread('cat_alone_sq.jpg');
dog1 = imread('dog_alone_sq.jpg');
%dog1 = data(:,:,:,157);
%cat1 = data(:,:,:,92);
if 0
ll=single(cat(2, labels{:}));
idx_dog=(find(ll==6));
for ix=idx_dog
    figure(1);imshow(data(:,:,:,ix)/255, 'InitialMagnification', 'fit')
    ix
    input('press key')
end 
idx_cat=(find(ll==4));
for ix=idx_cat
    figure(1);imshow(data(:,:,:,ix)/255, 'InitialMagnification', 'fit')
    ix
    input('press key')
end 
end
itx = 1;
max_tx = 10;
min_tx = -10;
steps_per_tx = 20/300;%Piyush
alpha_steps = 1/300; %Piyush
tx_range = min_tx:steps_per_tx:max_tx;
alpha_range=0:alpha_steps:1;
ty = 0;
data_vid = [];
total = 1;

for alpha = alpha_range
    dogcat1 = cat1*alpha + (1-alpha)*dog1;%Piyush
    outimg = imresize(dogcat1,[32, 32]);
    for tx=tx_range
        outimg_shifted = imtranslate(outimg,[tx, ty],'FillValues',255);
        data_vid(:,:,:,total ) = outimg_shifted;
        tx_used(total ) = tx;
        alpha_used(total ) =  alpha;        
        %fprintf('.')
        %imshow(outimg_shifted)
        total = total + 1;
    end 
    itx = itx + 1;      
    fprintf('\n %d : %d', itx, total);
end

data = bsxfun(@minus, single(data_vid), dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng
num_images = size(data,4);
set = 3*ones(1,num_images);
set(end) = 1;
labels = ones(1,num_images);

if opts.contrastNormalization
  z_orig = reshape(data_orig,[],60000) ;
  z = reshape(data,[],num_images) ;
  z_orig = bsxfun(@minus, z_orig, mean(z_orig,1)) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n_orig = std(z_orig,0,1) ;
  n = std(z,0,1) ;
  z_orig = bsxfun(@times, z_orig, mean(n_orig) ./ max(n_orig, 40)) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
  data_orig = reshape(z_orig, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],num_images) ;
  z_orig = reshape(data_orig,[],60000) ;
  W = z_orig(:,set == 1)*z_orig(:,set == 1)'/60000;  
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = labels;%single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
imdb.images.tx_used = tx_used;
imdb.images.alpha_used = alpha_used;


% -------------------------------------------------------------------------
function imdb = getCifarImdbEval3(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));

for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));
data_orig = data;

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
cat1 = imread('cat_alone_sq.jpg');
dog1 = imread('dog_alone_sq.jpg');
%dog1 = data(:,:,:,157);
%cat1 = data(:,:,:,92);
if 0
ll=single(cat(2, labels{:}));
idx_dog=(find(ll==6));
for ix=idx_dog
    figure(1);imshow(data(:,:,:,ix)/255, 'InitialMagnification', 'fit')
    ix
    input('press key')
end 
idx_cat=(find(ll==4));
for ix=idx_cat
    figure(1);imshow(data(:,:,:,ix)/255, 'InitialMagnification', 'fit')
    ix
    input('press key')
end 
end
itx = 1;
%max_tx = 15;
%min_tx = -15;
%steps_per_tx = 0.1;%Piyush
theta_min=0;
theta_max=360;
theta_steps = 360/360; %Piyush
alpha_steps = 1/300;
alpha_min = 0;
alpha_max = 1;
alpha_range = alpha_min:alpha_steps:alpha_max;
%tx_range = min_tx:steps_per_tx:max_tx;
theta_range = theta_min:theta_steps:theta_max;
%tx_used = zeros(1,numel(tx_range));
ty = 0;
data_vid = [];
total = 1;


for theta=theta_range        
    tform = affine2d([cosd(theta) sind(theta) 0;...
        -sind(theta) cosd(theta) 0; 1 1 1]);    
    
    dog_temp = imresize(dog1,[32, 32]);
    cat_temp = imresize(cat1, [32,32]);

    [dog_temp2] = imwarp(dog_temp,tform, 'FillValues', 255);      
    [cat_temp2] = imwarp(cat_temp,tform, 'FillValues', 255);  
    
    dog_temp3 = dog_temp2(1:32,1:32,:);
    cat_temp3 = cat_temp2(1:32,1:32,:);
    
    for alpha = alpha_range
        dogcat1 = cat_temp3*alpha + (1-alpha)*dog_temp3;%Piyush                        
        
        data_vid(:,:,:,total ) = dogcat1;
        
        theta_used(total ) = theta;
        alpha_used(total ) =  alpha;        
        %fprintf('.')
        %imshow(dogcat1)
        total = total + 1;
    end 
    itx = itx + 1;      
    fprintf('\n %d : %d', itx, total);
end

data = bsxfun(@minus, single(data_vid), dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng
num_images = size(data,4);
set = 3*ones(1,num_images);
set(end) = 1;
labels = ones(1,num_images);

if opts.contrastNormalization
  z_orig = reshape(data_orig,[],60000) ;
  z = reshape(data,[],num_images) ;
  z_orig = bsxfun(@minus, z_orig, mean(z_orig,1)) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n_orig = std(z_orig,0,1) ;
  n = std(z,0,1) ;
  z_orig = bsxfun(@times, z_orig, mean(n_orig) ./ max(n_orig, 40)) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
  data_orig = reshape(z_orig, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],num_images) ;
  z_orig = reshape(data_orig,[],60000) ;
  W = z_orig(:,set == 1)*z_orig(:,set == 1)'/60000;  
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = labels;%single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
imdb.images.alpha_used = alpha_used;
imdb.images.theta_used = theta_used;
