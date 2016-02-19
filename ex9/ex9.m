
function [ ] = ex9()
    % Linear SVM
    load('SVM_linear_data.mat');
    m = size(X, 1);
    
    w = soft_SVM(X, Y, 0.1, 10 * m);
    figure(1);
    show_SVM_linear(X, Y, w);
    title('Linear SVM');
    
    % Gaussian SVM
    load('SVM_gaussian_data.mat');
    m = size(X, 1);
    
    % sigma2 = 0.1
    alphas = soft_SVM_gaussian(X, Y, 0.1, 0.1, 10 * m);
    figure(2);
    show_SVM_gaussian(X, Y, alphas, 0.1);
    title('Gaussian SVM, \sigma^2=0.1');
    
    % sigma2 = 1
    alphas = soft_SVM_gaussian(X, Y, 0.1, 1, 10 * m);
    figure(3);
    show_SVM_gaussian(X, Y, alphas, 1);
    title('Gaussian SVM, \sigma^2=1');
    
    % sigma2 = 10
    alphas = soft_SVM_gaussian(X, Y, 0.1, 10, 10 * m);
    figure(4);
    show_SVM_gaussian(X, Y, alphas, 10);
    title('Gaussian SVM, \sigma^2=10');
end

function [ alphas ] = soft_SVM_gaussian( X, Y, lambda, sigma2, T )
%SOFT_SVM_GAUSSIAN algorith implementation
    % X is an mxd matrix, whose rows correspond to the instances
    % Y is an mx1 matrix, where Y_i is the label of X_i (either 1 or -1)
	% lambda is the regularization parameter.
	% T represents the number of iterations. 
    
	% The  output,  denoted w, is a dx1 vector, 
    % which is obtained by the soft-SVM algorithm
    m = size(X, 1);
    
    Z = X * X';
    D = repmat(diag(Z, 0), 1, m);
    G = exp(-(D + D' - 2 * Z) / sigma2);
    
    beta = zeros(m, 1);
    alphas = zeros(m, 1);
    for t = 1 : T
        alpha = (1/(lambda*t)) * beta;
        i = randi(m, 1);
        if Y(i) * dot(alpha, G(:,i)) < 1
            beta(i) = beta(i) + Y(i);
        end
        
        alphas = alphas + alpha;
    end
    
    alphas = alphas / T;

end
