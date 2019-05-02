function [ w2, b2,time2 ] = PinLTWSVM2( xA, xB, C2,V2,T2 )

[N1,D]=size(xA);
[N2,D]=size(xB);


H=[xA; zeros(N1,D)];
G=[xB; zeros(N2,D)];

% Quadratic term objective
obj_quad=G*G';
obj_quad=obj_quad+eps*eye(size(obj_quad)); %Conditioning
obj_quad=(obj_quad+obj_quad')/2;

% Linear term objective
obj_linear= G*H';
obj_linear=(-V2/N1)*sum(obj_linear,2);


% Setup inwquality constraints
A_ineq_const=[[-1*eye(N2),-1*eye(N2)];[zeros(N2,N2),-1*eye(N2)]];
b_ineq_const=zeros(2*N2,1);

% Setup equality constraints
A_eq_const=[eye(N2),(1+1/T2)*eye(N2)];
A_eq=[ones(1,N2),zeros(1,N2)];
A_eq_const=[A_eq_const;A_eq];
b_eq_const=(C2/N2)*ones(N2,1);
b_eq_const=[b_eq_const;V2];
% Setup options

% Solve QPP
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');
tic;
[X, FVAL, EXITFLAG,output]=quadprog(obj_quad, obj_linear, A_ineq_const, b_ineq_const, A_eq_const, b_eq_const,[],[],[],options);
time2=toc;
% Compute solution
lambda=X(1:N2,:);
%Calculating weights and bias term
w2=lambda'*xB-sum((V2/N1)*xA,1);
b2=-sum(xB*w2')/N2;

end

