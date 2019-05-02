function [ wB, bB ,time2] = LTWSVM2( xA, xB, C2 )

[N1,D]=size(xA);
[N2,D]=size(xB);


P=[xA,ones(N1,1)];
Q=[xB,ones(N2,1)];

alpha0=[rand(N1,1)];

% Quadratic term objective
obj_quad=P*pinv(Q'*Q+eps*eye(size(Q'*Q)))*P';
obj_quad=obj_quad+eps*eye(size(obj_quad)); %Conditioning
obj_quad=(obj_quad+obj_quad')/2; %Making symmetric

% Linear term objective
obj_linear=-ones(size(alpha0,1),1);

% Setup inwquality constraints
A_ineq_const=[];
b_ineq_const=[];

% Setup equality constraints
A_eq_const=[];
b_eq_const=[];

% Setup bounds
lb=zeros(size(alpha0,1),1);
ub=C2*ones(size(alpha0,1),1);

% Setup options
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');

% Solve QPP
tic;
[X, FVAL, EXITFLAG]=quadprog(obj_quad, obj_linear, A_ineq_const, b_ineq_const, A_eq_const, b_eq_const, lb, ub, [], options);
time2=toc;
% Compute solution
u=-pinv(Q'*Q + eps*eye(size(Q'*Q)))*P'*X;

wB=u(1:end-1,:);
bB=u(end,:);
    
end

