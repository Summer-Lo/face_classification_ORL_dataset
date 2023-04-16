clear;
% Loading the ORL dataset from Local drive
data_path = 'D:\01-PolyU-MSc\EIE522\Lab2\version2\face database\ORL';
num_subjects = 40;
num_images_per_subject = 10;
image_size = [92, 112];     % Check by testing program
reshape_size = 0.25;
% Generating the random number for split the data into training and testing
% dataset, which is required in the lab sheet.
train_num = 5;
test_num = 5;
rand_num = randperm(10);        % Generating random index number from 1 - 10 without repeating
train_order = rand_num(1:train_num);
test_order = rand_num((10-train_num+1):10);
trainIndex = rand_num(1:5);     % Checking the random number (Training data)
testIndex = rand_num(6:10);     % Checking the random number (Testing data)

train_data = [];
test_data = [];
train_label = [];
test_label = [];

for subject = 1:num_subjects
    for image_idx = 1:num_images_per_subject
        % Load the image
        image_path = fullfile(data_path, ['s', num2str(subject)], [num2str(image_idx), '.pgm']);
        image = imread(image_path);
        
        % Resize the image to the desired size
        image = imresize(image, reshape_size);
        
        % Split the data into training and testing sets
        if ((any(train_order==image_idx)==1))
            test_data = [test_data, double(image(:))];
            test_label = [test_label; subject];
        else % odd images for training
            train_data = [train_data, double(image(:))];
            train_label = [train_label; subject];
        end
    end
end

% Use fastPCA to reduce the dimensionality of the data
% fastpca download from paper of Joural of Physics: Conference Series
% Paper: Design and Realization of MATLAB-based Face Recognition System
% Cite: JianMing Liu 2018 J. Phys.: Conf. Ser. 1087 062033
k = 100; % number of principal components to keep
[train_data_PCA,V] = fastPCA(train_data', k);

% Project the data onto the principal components
train_data_pca = (train_data' * V)';
test_data_pca = (test_data' * V)';

% Normalize the data to have zero mean and unit variance
train_data_norm = zscore(train_data_pca, [], 2);
test_data_norm = zscore(test_data_pca, [], 2);

% save the calucated training dataset and testing dataset
save('face_dataset.mat', 'train_data_norm','train_label', 'test_data_norm', 'test_label');

% Restore data for model training
train_data = train_data_norm;
test_data = test_data_norm;
train_label = train_label;
test_label = test_label;

% Set up the neural network
num_input_nodes = size(train_data, 1);
num_hidden_nodes = 100;
num_output_nodes = length(unique(train_label));
one_hot_train_label = bsxfun(@eq, train_label(:), unique(train_label)');
%net = feedforwardnet(num_hidden_nodes);
net = patternnet(num_hidden_nodes);
% Train Function can be 'trainlm', 'trainbr', 'trainbfg', 'trainrp',
% 'trainscg', 'traincgb', 'traincgf', 'traincgp', 'trainoss', 'traingdx',
% 'traingdm', 'traingd'
% For analysis (2) of report, I used 'trainlm', 'traingdx', 'trainrp',
% 'trainbr', and 'traincgb' for comparing
net.trainFcn = 'trainlm';

% Specify the transfer functions for the hidden and output layers
net.layers{1}.transferFcn = 'logsig'; % sigmoidal transfer function
net.layers{2}.transferFcn = 'softmax'; % softmax transfer function

% Train the network using the BP algorithm
net.trainParam.lr = 0.01;
net.trainParam.epochs = 1000;
net = train(net, train_data, one_hot_train_label');

% Test the network
predicted_label = net(test_data);
[~, predicted_label] = max(predicted_label);
accuracy = sum(predicted_label' == test_label) / length(test_label)

