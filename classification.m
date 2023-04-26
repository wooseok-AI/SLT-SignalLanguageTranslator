%load data
filename = "test\construct.mp4";
%load trained model
model = load('bilstm_best.mat').netLSTM;
%load model for making sequence
resnet = resnet101;
layerName = 'pool5';
inputSize = resnet.Layers(1).InputSize(1:2);

%make sequence
video = readVideo(filename);
video = centerCrop(video,inputSize);
video = activations(resnet,video,layerName,'OutputAs','columns');
%classification
YPred = classify(model,{video})