%######################################
%
%######################################
imHeight = 159;
imWidth = 159;
epsilon = 1e-6;
imageNum = 165; % number of images
h = 15; % number of people
w = 11; % number of different expression according a single person

%rawImages = readyalefaces();
%centeredFaces = centerfaces(rawImages);

load('centeredFace.mat');
trainingExamples = getTrainingExample(centeredFaces, 13, w);
meanFace = meanface(trainingExamples, imHeight, imWidth);

% figure
% imagesc(meanFace);
% truesize;
% colormap(gray)

% subMeanFace is a (N by N) by 165 matrix which stores every faces in a column 
% after subtracting the meanFace. subFace is a N by N matrix representing a single 
% subMeanface.

[subMeanFace, subFace] = getSubMeanFace(trainingExamples, meanFace,...
                                                imHeight, imWidth, 13, w);

% figure
% imagesc(subFace(:,:,2));
% truesize;
% colormap(gray)

% compute eigenvalues and eigenvectors of A' * A (A = subMeanFace)
[U, S, V] = svd(subMeanFace' * subMeanFace);
%fprintf('size of S: (%i, %i)', size(S));

% implement PCA whitening on A
% and compute eigenvectors of A*A', or to say, eigenfaces
eigenVecs = subMeanFace * U;
norm = sqrt(cumsum(eigenVecs(:, 1:end) .* eigenVecs(:, 1:end)));
eigenFaces = bsxfun(@rdivide, eigenVecs, norm(end, :));

% get reshaped eigenfaces and K of them with corresponding largest eigenvalues
[kReshapedEigenFaces, kChosenEigenFaces] = getReshapedEigenFace(eigenFaces,...
                                                                imHeight, imWidth, 13, w, 20);
%reshapedEigenFaces = getReshapedEigenFace(eigenFaces, imHeight, imWidth, imageNum);
%tightsubplot(15, 11, reshapedEigenFaces);
%displayEigenFace(kReshapedEigenFaces);

% calculate the weights and then normalize them
eigenWeight = getWeight(kChosenEigenFaces, subMeanFace);
%eigenWeight = eigenWeight ./ max(abs(eigenWeight(:)))

testExample = loadTestExample(centeredFaces, trainingExamples, 8);
testExample = testExample - meanFace(:);

testExProj = kChosenEigenFaces' * testExample;

nearerstNeigb = knnSearch(eigenWeight, testExProj, 10);

%labeledFace = setFaceLabel(trainingExamples, h, w);
