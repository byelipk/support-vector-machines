function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

vals      = [ 0.01 0.03 0.1 0.3 1 3 10 30 ]';
C_vec     = vals;
sigma_vec = vals;
results   = zeros(64, 3);

% Create a grid of all possible combinations of C and Sigma
[p, q] = meshgrid(C_vec, sigma_vec);
pairs  = [p(:), q(:)];

for row = [1:64],
  printf('ITERATION %d\n\n', row);

  c = pairs(row, 1);
  s = pairs(row, 2);

  % Train the model
  model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));

  % Run the predictions
  preds = svmPredict(model, Xval);

  % Compute the cross validation error
  error = mean(double(preds ~= yval));

  % Save the results
  results(row, 1) = c;
  results(row, 2) = s;
  results(row, 3) = error;
end;

fprintf('\n\n');
fprintf('Table of Results\n');
fprintf('C\t\tSigma\t\tError\n');
for i = 1:64,
	fprintf(' %f\t%f\t%f\n', ...
            results(i, 1), results(i, 2), results(i, 3));
end
printf('\n\n');

[error, index] = min(results(:, 3));
C = results(index, 1);
sigma = results(index, 2);

fprintf('C\t\tSigma\t\tMin Error\n');
fprintf(' %f\t%f\t%f\n', C, sigma, error);

% =========================================================================

end
