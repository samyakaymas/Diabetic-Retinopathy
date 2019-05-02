function [ w, b,time ] = SVM(xtrain, ytrain, C)
[M,N]=size(xtrain);
y=ytrain*ones(1,N);
H=xtrain.*(1.0*y);
H=H*H';
%
obj_quad=H;
%Conditioning
obj_quad=obj_quad+eps*eye(size(obj_quad)); 
obj_quad=(obj_quad+obj_quad')/2;
%Q in quadratic linear equation
obj_linear=-ones(M,1);
%Inequality constraints
A_ineq_const=[-eye(M);eye(M)];
b_ineq_const=[zeros(M,1);C*ones(M,1)];

%Equality constraints
A_eq_const=1.0*ytrain';
b_eq_const=0.0;
% Setup options
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');

% Solve QPP
tic;
[X, FVAL, EXITFLAG]=quadprog(obj_quad, obj_linear, A_ineq_const, b_ineq_const, A_eq_const,b_eq_const,0*ones(M,1),C*ones(M,1),[],options);
time=toc;
w=xtrain.*((X.*ytrain)*ones(1,N));
w=sum(w,1);
S=(X>eps);
b=xtrain*w';
b=ytrain(S)-b(S);
b=sum(b)/length(b);
end