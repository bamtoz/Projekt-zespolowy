%RESNET

%lokalizacja plików
outputFolderR = fullfile('DOBRE_DataSet');
rootFolderR = fullfile(outputFolder, 'Resnet224');

%podzia³ na kategorie i odpowiednia ilosc zdjec
categoriesR = {'Normal','Cancer'};
imdsR = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
countEachLabel(imdsR) 

%tworzenie sieci
netR = resnet50();
figure
plot(netR)
title('First section of ResNet-50')
set(gca,'YLim',[150 170]);

numel(netR.Layers(end).ClassNames)

podzial na dane do uczenia i testowe w stosunku 30:70
[trainingSetR, testSetR] = splitEachLabel(imdsR, 0.3, 'randomize');

przygotowanie sieci do uczenia
imageSizeR = netR.Layers(1).InputSize;
augmentedTrainingSetR = augmentedImageDatastore(imageSizeR, trainingSetR, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSetR = augmentedImageDatastore(imageSizeR, testSetR, 'ColorPreprocessing', 'gray2rgb');
w1R = netR.Layers(2).Weights;
w1R = mat2gray(w1R);
w1R = imresize(w1R,5); 
featureLayerR = 'fc1000';
trainingFeaturesR = activations(netR, augmentedTrainingSetR, featureLayerR, 'MiniBatchSize', 32, 'OutputAs', 'columns');
trainingLabelsR = trainingSetR.Labels;
classifierR = fitcecoc(trainingFeaturesR, trainingLabelsR, 'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
testFeaturesR = activations(netR, augmentedTestSetR, featureLayerR, 'MiniBatchSize', 32, 'OutputAs', 'columns');

wyniki
predictedLabelsR = predict(classifierR, testFeaturesR, 'ObservationsIn', 'columns');
testLabelsR = testSetR.Labels;

confMatR = confusionmat(testLabelsR, predictedLabelsR);
confMatR = bsxfun(@rdivide,confMatR,sum(confMatR,2))
mean(diag(confMatR))

% 
% %%% TEST %%%
% testImage = imread(fullfile('benign.tif'));
% 
% % Create augmentedImageDatastore to automatically resize the image when
% % image features are extracted using activations.
% ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');
% 
% % Extract image features using the CNN
% imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');
% 
% % Make a prediction using the classifier
% predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')
% 
% % Display predicted label
% sprintf('The loaded image belongs to %s class',label)
 