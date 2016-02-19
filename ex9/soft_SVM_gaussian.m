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
