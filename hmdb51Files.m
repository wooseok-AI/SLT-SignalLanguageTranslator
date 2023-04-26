function [files, labels] = hmdb51Files(dataFolder)
%hmdb51Files List of files and labels from the HMDB dataset
%   [files, labels] = hmdb51Files(dataFolder) returns a list of files and
%   labels from the HMDB dataset given by dataFolder

fileExtension = ".mp4";
listing = dir(fullfile(dataFolder, "*", "*" + fileExtension));

numObservations = numel(listing);
files = strings(numObservations,1);
labels = cell(numObservations,1);

for i = 1:numObservations
    name = listing(i).name;
    folder = listing(i).folder;
    
    [~,labels{i}] = fileparts(folder);
    files(i) = fullfile(folder,name);
end

labels = categorical(labels);

end
