function [net, info] = cnn_mnist(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.mnistVer = 5;
opts.imdbEval = 1;
opts.batchNormalization = false ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['mnist-baseline-' num2str(opts.mnistVer) sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.imdbPathEval = fullfile(opts.expDir, 'imdb_eval.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
    if  opts.mnistVer == 1
        net = cnn_mnist_init('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType) ;
    elseif opts.mnistVer == 2
        net = cnn_mnist_init2('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType) ;
    elseif opts.mnistVer == 3
        net = cnn_mnist_init3('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType) ;
    elseif opts.mnistVer == 4
        net = cnn_mnist_init4('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType) ;
    elseif opts.mnistVer == 5
        net = cnn_mnist_init5('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType) ;

    end 
        

else
  net = opts.network ;
  opts.network = [] ;
end

if opts.imdbEval == 0
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
else
if exist(opts.imdbPathEval, 'file')
  imdb = load(opts.imdbPathEval) ;
else
  imdb = getMnistImdbEval(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPathEval, '-struct', 'imdb') ;
end
end 


net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3),...
  'mnistVer', opts.mnistVer) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));

if opts.mnistVer ~= 1
	labels = cat(2, y1, y2);
	valid_label_idx = find(labels == 7 | labels == 10);
	data = data(:,:,:,valid_label_idx);
	set = set(valid_label_idx);
	valid_labels = labels(valid_label_idx);
	valid_labels(find(valid_labels == 7)) = 1;
	valid_labels(find(valid_labels == 10)) = 2;
end 
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;


imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
if opts.mnistVer ~= 1
	imdb.images.labels = valid_labels;
end 
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;


% --------------------------------------------------------------------
function imdb = getMnistImdbEval(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

img = data(:,:,numel(y1)+13);
finlabel = 10;% imdb.images.labels(numel(y1)+13);
min_deg = 1;
max_deg=360;
steps_per_deg=1;
max_tx = 15;
min_tx = -15;
steps_per_tx = 0.1;%
tx_range = min_tx:steps_per_tx:max_tx;
theta_range = min_deg:steps_per_deg:max_deg;
outdata = zeros(28,28,1,numel(theta_range)*numel(tx_range));
scale = 1;
itx = 0;
HIRES = 2048;
PAD = 256;
%img = imresize(img,[HIRES,HIRES]);
%img = padarray(img,[PAD PAD],0,'both');
theta_used = [];
tx_used = [];
ty = 0;
for theta = theta_range
for tx = tx_range
    itx = itx + 1;
    tform = affine2d([cosd(theta) sind(theta) 0;...
    -sind(theta) cosd(theta) 0; 1 1 1]);
%    img_ref = imref2d(size(img));
%    cb_translated_ref = img_ref;
%    cb_translated_ref.XWorldLimits(2) = cb_translated_ref.XWorldLimits(2)+tx;
%    cb_translated_ref.YWorldLimits(2) = cb_translated_ref.YWorldLimits(2)+0;
%    [outimg,cb_translated_ref] = imwarp(img,tform,'OutputView',cb_translated_ref);
    [outimg] = imwarp(img,tform);
    outimg = imtranslate(outimg,[tx, ty],'FillValues',0);
    %outimg = img;
    %outimg = imcrop(outimg,[PAD+tx,PAD, HIRES, HIRES]);
    outdata(:,:,1,itx) = imresize(outimg,[28, 28]);
    tx_used(itx) = tx;
    theta_used(itx) = theta;
    imshow(imresize(outdata(:,:,itx), [128 128]));
end 
theta=theta
end 
outdata=cat(4,outdata,zeros(28,28,1,1));
imdb.images.data = single(outdata);
sz = size(outdata);
imdb.images.labels = 2*ones(1,sz(4),'single');
imdb.images.set = 3*ones(1,sz(4),'single');
imdb.images.set(end) = 1;
imdb.images.tx_used = tx_used;
imdb.images.theta_used = theta_used;
