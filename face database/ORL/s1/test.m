function [imgRow,imgCol,FaceContainer,faceLabel] = ReadFaces (nFacePerPerson,nPerson,bTest)
    if nargin==0
        nFacesPerPerson=5;
        nPerson=40;
        bTest=0;
    elseif nargin<3
        bTest=0;
    end
    img = imread(strcat('P:\EIE522\Lab2\face database\ORL\s1\1.pgm'));
    [imgRow,imgCol] = size(img);
    randNo = randperm(10,10);
    trainIndex = randNo(1:5);
    testIndex = randNo(6:10);
    for i=1:nPerson
        i0=char(i/10);  %i0 in range of 
end