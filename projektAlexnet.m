%ALEXNET

%lokalizacja plików
outputFolder = fullfile('DOBRE_DataSet');
rootFolder = fullfile(outputFolder, 'Alexnet227');

%podzia³ na kategorie i odpowiednia ilosc zdjec
categories = {'Normal','Cancer'};
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
countEachLabel(imds) 
 
%tworzenie sieci
net = alexnet;

figure
plot(net)
title('First section of AlexNet')
set(gca,'YLim',[150 170]);


numel(net.Layers(end).ClassNames);

%podzial na dane do uczenia i testowe w stosunku 30:70
[trainingSet,testSet] = splitEachLabel(imds,0.3, 'randomize');

%przygotowanie sieci do uczenia
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet,'ColorPreprocessing','gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet,'ColorPreprocessing','gray2rgb');
w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
featureLayer  = 'fc7';
trainingFeatures = activations(net,augmentedTrainingSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');
trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learner','Linear','Coding','onevsall','ObservationsIn','columns');
testFeatures = activations(net,augmentedTestSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

%wyniki
predictLabels = predict(classifier, testFeatures,'ObservationsIn','columns');
testLabels = testSet.Labels;

confMat = confusionmat(testLabels,predictLabels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
meanAccuracy = mean(diag(confMat));
 
% 
% 
% % %%% TEST %%%
% % %  
% % % newImage = imread(fullfile('normal.tif'));
% % % 
% % % ds = augmentedImageDatastore(imageSize, newImage,'ColorPreprocessing','gray2rgb');
% % % 
% % % imageFeatures = activations(net,ds,featureLayer,'MiniBatchSize',32,'OutputAs','columns');
% % % 
% % % label = predict(classifier, imageFeatures,'ObservationsIn','columns');
% 
% % % sprintf('The loaded image belongs to %s class',label)
