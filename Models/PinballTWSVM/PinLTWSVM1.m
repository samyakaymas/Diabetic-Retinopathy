function [ w1, b1 ,time1] = PinLTWSVM1( xA, xB, C1,V1,T1 )

[N1,D]=size(xA);
[N2,D]=size(xB);


H=[xA; zeros(N1,D)];
G=[xB; zeros(N2,D)];

% Quadratic term objective
obj_quad=H*H';
obj_quad=obj_quad+eps*eye(size(obj_quad)); %Conditioning
obj_quad=(obj_quad+obj_quad')/2;

% Linear term objective
obj_linear= H*G';
obj_linear=(-V1/N2)*sum(obj_linear,2);


% Setup inwquality constraints
A_ineq_const=[[-1*eye(N1),-1*eye(N1)];[zeros(N1,N1),-1*eye(N1)]];
b_ineq_const=zeros(2*N1,1);

% Setup equality constraintsj
A_eq_const=[eye(N1),(1+1/T1)*eye(N1)];
A_eq=[ones(1,N1),zeros(1,N1)];
A_eq_const=[A_eq_const;A_eq];
b_eq_const=(C1/N1)*ones(N1,1);
b_eq_const=[b_eq_const;V1];
% Setup options

% Solve QPP
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');
tic;
[X, FVAL, EXITFLAG]=quadprog(obj_quad, obj_linear, A_ineq_const, b_ineq_const, A_eq_const, b_eq_const,[],[],[],options);
time1=toc;
% Compute solution
lambda=X(1:N1,:);
%Calculating weights and bias term
w1=lambda'*xA-sum((V1/N2)*xB,1);
b1=-sum(xA*w1')/N1;
end

