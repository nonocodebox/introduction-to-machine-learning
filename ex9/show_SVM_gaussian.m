function  [] = show_SVM_gaussian(X,Y,alphas,sigma2)
% Given data (X,Y) and a kernel classifier alpha, display the result.

plot(X(Y==1,1),X(Y==1,2),'+','markeredgecolor','b','markersize',10,'linestyle','none','linewidth',3);hold on;
plot(X(Y==-1,1),X(Y==-1,2),'o','markeredgecolor','r','markersize',10,'linestyle','none','linewidth',3);
minx = min(X(:,1));
maxx = max(X(:,1));
miny = min(X(:,2));
maxy = max(X(:,2));
X_range = minx:0.05:maxx;
Y_range = miny:0.05:maxy;
Z = zeros(length(X_range),length(Y_range));
axis([minx maxx miny maxy]);
m = size(X,1);
for i=1:length(X_range)
    for j=1:length(Y_range)
        Z(i,j) = alphas'*exp(-sum((repmat([X_range(i) Y_range(j)],m,1)-X).^2,2)/sigma2);
    end;
end;
[A B] = meshgrid(Y_range,X_range);
contour(B,A,Z,[0 0],'k'); hold off;


