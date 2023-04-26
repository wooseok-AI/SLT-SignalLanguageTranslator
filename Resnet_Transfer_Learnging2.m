%load resnet
resnet = resnet50;
%load train data
dataFolder = "train_data";
fprintf("Storing Train Data\n")
[files, labels] = hmdb51Files(dataFolder);
inputSize = resnet.Layers(1).InputSize(1:3);
numFiles = numel(files);
data = cell(numFiles,1);

for i =1:numel(files)
   
    data{i,1} = video;
end

resnet = resnet50;
%Train Test Split
fprintf("Train Test Split\n")
numObservations = numel(data);
idx = randperm(numObservations);
N = floor(0.7 * numObservations);

idxTrain = idx(1:N);
dataTrain = data(idxTrain);
labelsTrain = labels(idxTrain);

idxValidation = idx(N+1:end);
dataValidation = data(idxValidation);
labelsValidation = labels(idxValidation);

%Transfer Learning
layersTransfer = resnet.Layers(1:end-3);
numFeatures = size(dataTrain{1},1);
numClasses = numel(categories(labelsTrain));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'Name','fc1','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name','sm1')
    classificationLayer('Name','cf1')];

miniBatchSize = 8;
numObservations = numel(dataTrain);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-4, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{dataValidation,labelsValidation}, ...
    'ValidationFrequency',numIterationsPerEpoch, ...
    'Plots','training-progress', ...
    'Verbose',false);



% layer 계층 연결 수정
lgraph = layerGraph(layers);

lgraph = disconnectLayers(lgraph,'res2a_branch2c', 'res2a_branch1');
lgraph = disconnectLayers(lgraph,'bn2a_branch1', 'add_1/in1');
lgraph = disconnectLayers(lgraph,'bn2a_branch2c', 'bn2a_branch1');
lgraph = disconnectLayers(lgraph,'res2a_branch1', 'bn2a_branch2c');
lgraph = connectLayers(lgraph,'max_pooling2d_1','res2a_branch1');
lgraph = connectLayers(lgraph,'res2a_branch2c','bn2a_branch2c');
lgraph = connectLayers(lgraph,'bn2a_branch2c','add_1/in1');
lgraph = connectLayers(lgraph,'res2a_branch1','bn2a_branch1');
lgraph = connectLayers(lgraph,'bn2a_branch1','add_1/in2');

lgraph = connectLayers(lgraph,'activation_4_relu','add_2/in2');
lgraph = connectLayers(lgraph,'activation_7_relu','add_3/in2');

lgraph = disconnectLayers(lgraph,'res3a_branch2c', 'res3a_branch1');
lgraph = disconnectLayers(lgraph,'bn3a_branch1', 'add_4/in1');
lgraph = disconnectLayers(lgraph,'bn3a_branch2c', 'bn3a_branch1');
lgraph = disconnectLayers(lgraph,'res3a_branch1', 'bn3a_branch2c');
lgraph = connectLayers(lgraph,'activation_10_relu','res3a_branch1');
lgraph = connectLayers(lgraph,'res3a_branch2c','bn3a_branch2c');
lgraph = connectLayers(lgraph,'bn3a_branch2c','add_4/in1');
lgraph = connectLayers(lgraph,'res3a_branch1','bn3a_branch1');
lgraph = connectLayers(lgraph,'bn3a_branch1','add_4/in2');


lgraph = connectLayers(lgraph,'activation_13_relu','add_5/in2');
lgraph = connectLayers(lgraph,'activation_16_relu','add_6/in2');
lgraph = connectLayers(lgraph,'activation_19_relu','add_7/in2');

lgraph = disconnectLayers(lgraph,'res4a_branch2c', 'res4a_branch1');
lgraph = disconnectLayers(lgraph,'bn4a_branch1', 'add_8/in1');
lgraph = disconnectLayers(lgraph,'bn4a_branch2c', 'bn4a_branch1');
lgraph = disconnectLayers(lgraph,'res4a_branch1', 'bn4a_branch2c');
lgraph = connectLayers(lgraph,'activation_22_relu','res4a_branch1');
lgraph = connectLayers(lgraph,'res4a_branch2c','bn4a_branch2c');
lgraph = connectLayers(lgraph,'bn4a_branch2c','add_8/in1');
lgraph = connectLayers(lgraph,'res4a_branch1','bn4a_branch1');
lgraph = connectLayers(lgraph,'bn4a_branch1','add_8/in2');

lgraph = connectLayers(lgraph,'activation_25_relu','add_9/in2');
lgraph = connectLayers(lgraph,'activation_28_relu','add_10/in2');
lgraph = connectLayers(lgraph,'activation_31_relu','add_11/in2');
lgraph = connectLayers(lgraph,'activation_34_relu','add_12/in2');
lgraph = connectLayers(lgraph,'activation_37_relu','add_13/in2');

lgraph = disconnectLayers(lgraph,'res5a_branch2c', 'res5a_branch1');
lgraph = disconnectLayers(lgraph,'bn5a_branch1', 'add_14/in1');
lgraph = disconnectLayers(lgraph,'bn5a_branch2c', 'bn5a_branch1');
lgraph = disconnectLayers(lgraph,'res5a_branch1', 'bn5a_branch2c');
lgraph = connectLayers(lgraph,'activation_40_relu','res5a_branch1');
lgraph = connectLayers(lgraph,'res5a_branch2c','bn5a_branch2c');
lgraph = connectLayers(lgraph,'bn5a_branch2c','add_14/in1');
lgraph = connectLayers(lgraph,'res5a_branch1','bn5a_branch1');
lgraph = connectLayers(lgraph,'bn5a_branch1','add_14/in2');

lgraph = connectLayers(lgraph,'activation_43_relu','add_15/in2');
lgraph = connectLayers(lgraph,'activation_46_relu','add_16/in2');

fprintf("Start Training\n")
[resnet,info] = trainNetwork(dataTrain,labelsTrain,lgraph,options);


%layer 연결 확인 
analyzeNetwork(lgraph)