function [ w ] = soft_SVM( X, Y, lambda, T )
%SOFT_SVM algotithm implementation
    % X is an mxd matrix, whose rows correspond to the instances
    % Y is an mx1 matrix, where Y_i is the label of X_i (either 1 or -1)
	% lambda is the regularization parameter.
	% T represents the number of iterations. 
    
	% The  output,  denoted w, is a dx1 vector, 
    % which is obtained by the soft-SVM algorithm
    m = size(X, 1);
    d = size(X, 2);
    theta = zeros(1, d);
    w_sum = zeros(1, d);
    
    for t = 1 : T
        w = (1/(lambda*t)) * theta;
        i = randi(m, 1);
        if Y(i) * dot(w, X(i, :)) < 1
            theta = theta + Y(i) * X(i, :);
        end
        
        w_sum = w_sum + w;
    end
    
    w = w_sum / T;
end
