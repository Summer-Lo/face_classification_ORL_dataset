% fastpca download from paper of Joural of Physics: Conference Series
% Paper: Design and Realization of MATLAB-based Face Recognition System
% Cite: JianMing Liu 2018 J. Phys.: Conf. Ser. 1087 062033
function [pcaA V] = fastPCA( A, k )
% Fast PCA
% Input: A ---sample matrix, each row as a sample
% k --- reduce dimension to Dimension k
% Output: pcaA --- Matrix consisting of Dimension k feature vector after dimension reduction,
%each row as a sample, Column k as sample feature dimension number after dimension reduction
% V --- primary component vector
[r c] = size(A);
% mean value of Samples
meanVec = mean(A);
% compute conversion of covariance matrix :cov Mat T
Z = (A-repmat(meanVec, r, 1));
covMatT = Z * Z';
% compute previous k eigenvalues and eigenvectors
[V D] = eigs(covMatT, k);
% extract eigenvectors of covariance matrix
V = Z' * V;
% normalize eigenvectors to unit eigenvectors
for i=1:k
 V(:,i)=V(:,i)/norm(V(:,i));
end
% Through linear transformation(projection), reduce dimension to Dimension k
pcaA = Z * V;
% save transformation matrix V and transformation origin mean Vec 
save('PCA.mat', 'V', 'meanVec');
end