function [meanVgg1,meanVgg2] = ExtractFCNfeature(matRoot,imname,pixelList,m,n)
layer1 = 32;
layer2 = 6;

padding = [0 100 100 100 100 ... 
            52 52 52 52 52 ... 
            26 26 26 26 26 26 26 ... 
            14 14 14 14 14 14 14 ...
            7 7 7 7 7 7 7 ...
            4];


matName1 = [matRoot,'32','/',imname,'.mat'];

eval(['load ',matName1,';']);
temp = layer32;    % feat
% data = load(matName1, '-ASCII');
% temp = data.layer32
%   temp = feat;       % PASCAL-S ʹ��
vgg_feat1 = temp(padding(layer1):end-padding(layer1)+1,padding(layer1):end-padding(layer1)+1,:);
disp(size(vgg_feat1))
vgg_feat1 = double(imresize(vgg_feat1,[m,n]));
meanVgg1 = GetMeanColor(vgg_feat1,pixelList,'vgg'); 

% Display dimensions of loaded variables
disp('Dimensions of variables from matName1:');
disp(['temp: ', num2str(size(temp))]);
disp(['vgg_feat1: ', num2str(size(vgg_feat1))]);
disp(['meanVgg1: ', num2str(size(meanVgg1))]);

matName2 = [matRoot,'6','/',imname,'.mat'];
eval(['load ',matName2,';']);
temp = layer6;
% data = load(matName2, '-ASCII');
% temp = data.layer6
%   temp = feat;         % PASCAL-S ʹ��
vgg_feat2 = temp(padding(layer2):end-padding(layer2)+1,padding(layer2):end-padding(layer2)+1,:);
disp(size(vgg_feat2))
vgg_feat2 = double(imresize(vgg_feat2,[m,n]));
meanVgg2 = GetMeanColor(vgg_feat2,pixelList,'vgg');

% Display dimensions of loaded variables
disp('Dimensions of variables from matName2:');
disp(['temp: ', num2str(size(temp))]);
disp(['vgg_feat2: ', num2str(size(vgg_feat2))]);
disp(['meanVgg2: ', num2str(size(meanVgg2))]);
