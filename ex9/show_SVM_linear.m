function  [] = show_SVM_linear(X,Y,w)
% Given data (X,Y) and a classifier w, display the result.

plot(X(Y==1,1),X(Y==1,2),'+','markeredgecolor','b','markersize',10,'linestyle','none','linewidth',3);hold on;
plot(X(Y==-1,1),X(Y==-1,2),'o','markeredgecolor','r','markersize',10,'linestyle','none','linewidth',3);
minx = min(X(:,1));
maxx = max(X(:,1));
miny = min(X(:,2));
maxy = max(X(:,2));
A = minx:0.01:maxx;
plot(A,-(w(1)/w(2))*A,'k'); hold off;
axis([minx maxx miny maxy]);
