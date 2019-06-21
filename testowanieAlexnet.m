%for indeks = 1:35
    indeks=1;
    indeks2 = num2str(indeks);
    file_name = strcat('33.tif');
    SourceIMG = imread(file_name);
    TargetIMG = imread('WZOR_WZOR.tif');
    [x,y,z] = size(SourceIMG);

    RGB_S = reshape(im2double(SourceIMG),[],3)';
    RGB_T = reshape(im2double(TargetIMG),[],3)';
         
    RGB_S = max(RGB_S,1/255);
    RGB_T = max(RGB_T,1/255);

    % Konwersja na LMS

    LMS_S = [0.3811 ,0.5783, 0.0402;
            0.1967, 0.7244, 0.0782;
            0.0241, 0.1288, 0.8444;]*RGB_S;

    LMS_T = [0.3811 ,0.5783, 0.0402;
            0.1967, 0.7244, 0.0782;
            0.0241, 0.1288, 0.8444;]*RGB_T;
 
   % Logarytm
    LMS_S = log10(LMS_S);
    LMS_T = log10(LMS_T);

    % LAB

    Lab_S = [ (1/sqrt(3)), 0 , 0; 0, (1/sqrt(6)), 0 ;0,0, (1/sqrt(2));] * [1,1,1;1,1,-2;1,-1,0]*LMS_S;
    Lab_T = [ (1/sqrt(3)), 0 , 0; 0, (1/sqrt(6)), 0 ;0,0, (1/sqrt(2));] * [1,1,1;1,1,-2;1,-1,0]*LMS_T;

    mean_s = mean(Lab_S,2); %  l*   -> srednia
    std_s = std(Lab_S,0,2); %l'   -> standard deviations
    mean_t = mean(Lab_T,2);
    std_t = std(Lab_T,0,2);

    sf = std_t./std_s;

    for ch = 1:3 
        Nowe_Lab(ch,:) = (Lab_S(ch,:) - mean_s(ch))*sf(ch) + mean_t(ch);
    end

    %%% Lab -> LMS

    Tab_pomoc =  [ (1/sqrt(3)), (1/sqrt(6)),(1/sqrt(2));
                (1/sqrt(3)),(1/sqrt(6)),(-1/sqrt(2));
                (1/sqrt(3)),(2/sqrt(6)),0];
    Nowe_LMS = Tab_pomoc*Nowe_Lab;

    for ch = 1:3
        Nowe_LMS(ch,:) = 10.^Nowe_LMS(ch,:);
    end
 
    %%% LMS -> RGB
 
    Nowe_RGB = ( [ 4.4679, -3.5873, 0.1193;
                -1.2186, 2.3809, -0.1624;
                0.0497, 0.2439, 1.2045]*Nowe_LMS )';
    Nowe_RGB = reshape(Nowe_RGB,size(SourceIMG));
    
    imwrite(Nowe_RGB,strcat('znormalizowane.tif'));

    file = dir ('znormalizowane.tif'); %Zawiera wszystkie pliki .tif

    destinationFolder = strcat(cd,'\SplittedA'); %Tworzy folder o nazwie splitted
    if ~exist(destinationFolder, 'dir')          % o ile nie bylo juz
        mkdir(destinationFolder);
    end

    filename2 = strcat('znormalizowane.tif');
    rgbImage = imread(filename2);

    %rows -> 1536, columns -> 2048, numberOfColorBands -> 3
    [rows, columns, numberOfColorBands] = size(rgbImage);

    %%Nasz wymagany rozmiar
    blockSizeR = 227;
    blockSizeC = 227;

    wholeBlockRows = floor(rows / blockSizeR); %
    blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];

    wholeBlockCols = floor(columns / blockSizeC);
    blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];

    %Macierz 7x10 elementow, zawiera tez elementy o wielkosci roznej niz
    %227x227, gdy oryginalny rozmiar nie byl calkowicie podzielny na 227
    ca = mat2cell(rgbImage, blockVectorR, blockVectorC, numberOfColorBands);
    
    %Ilosc calkowitych kwadratow
    numPlotsR = wholeBlockRows; 
    numPlotsC = wholeBlockCols;
    
    wektor_wynikow = zeros(wholeBlockRows,wholeBlockCols);
    
    saveIndex = 1;
    for r = 1 : wholeBlockRows
       for c = 1 : wholeBlockCols
           splittedImage = ca{r,c};
           imwrite(splittedImage, fullfile(cd, 'SplittedA', sprintf('output%d.tif',saveIndex)));
           ds = augmentedImageDatastore(imageSize, splittedImage,'ColorPreprocessing','gray2rgb');
           imageFeatures = activations(net,ds,featureLayer,'MiniBatchSize',32,'OutputAs','columns');
           label = predict(classifier, imageFeatures,'ObservationsIn','columns');
       %    sprintf('The loaded image belongs to %s class',label)
           wektor_wynikow(r,c) = label;
           saveIndex = saveIndex + 1;
       end
    end
    
    srednia = mean(wektor_wynikow, 'all')
    
    if(srednia>1.5)
           sprintf('The loaded image nr %d belongs to Normal class',indeks)
    else
           sprintf('The loaded image nr %d  belongs to Cancer class',indeks)
    end
    
%end
