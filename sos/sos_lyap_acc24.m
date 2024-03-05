clc; clear all; close all

sdpvar x1 x2 

example = 10;

% [1.25, 2.4, 1.96, 1.7, 0.81, 0.81, 0.62, 2.23, 1.7, 1.92]

switch example
    case 1
        mu = 1.25;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_1";
        deg_f = 3;
    case 2
        mu = 2.4;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_2";        
    case 3
        mu = 1.96;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_3";   
    case 4
        mu = 1.7;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_4";       
    case 5
        mu = 0.81;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_5";       
    case 6
        mu = 0.81;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_6";       
    case 7
        mu = 0.62;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_7";   
    case 8
        mu = 2.23;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_8";       
    case 9
        mu = 1.7;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "vdp_network_subsystem_9";       
    case 10
        mu = 1.92;
        f = [-x2; x1-mu*x2*(1-x1^2)];
        A = [0 -1;1 -mu];
        name = "van_der_pol_subsystem_10";  
    otherwise
        error('Invalid example selection');
end

% Make the code work for examples of different dimensions;
d = length(f);

x = [];
for i = 1:d
    x = [x; eval(['x' num2str(i)])];
end

% Adjust the degrees of the SOS polynomials accordingly
% Requirements are: 
% deg(s2*V)>=deg(dV*f*s3) and deg(h*s1)>=deg(V)
deg_s1 = 4;
deg_s2 = 4;
%deg_s3 = 4;
deg_V = 6;

% Initialize to Quadratic Lyapunov Function
P = lyap(A',eye(d));
V = x'*P*x;
dVdt = jacobian(V,x)*f;

% Precision parameters
bisection_precision = 1e-3;
stop_precision = 1e-4;

% Lower bound and shape functions
l = 1e-6*sum(x.^2);
h = sum(x.^2);

% SOS conditions and SDP solver
ops = sdpsettings('solver','mosek', 'verbose', 0);
% ops = sdpsettings('solver','SeDuMi', 'verbose', 0);
% ops = sdpsettings('solver','sdpt3', 'verbose', 0);

V_last_feasible = [];
beta_last_feasible = [];
V_infeasible = 0;

tic;

%% gamma step 
lower_bound = 0;
upper_bound = 10; 
while (upper_bound - lower_bound) > bisection_precision
    gamma = (lower_bound + upper_bound) / 2;    
    [s2,c2] = polynomial(x,deg_s2);
%    [s3,c3] = polynomial(x,deg_s3);
    % F = [sos(s2), sos(-(l + dVdt*s3) + s2*(V - gamma))];
    F = [sos(s2), sos(-(l + dVdt) + s2*(V - gamma))];
    sol = solvesos(F,[],ops,c2);
    if sol.problem == 0
        lower_bound = gamma;
    else
        upper_bound = gamma;
    end
end
gamma = lower_bound

V = V/gamma; % normalize V
gamma = 1.0;

%% beta step 
lower_bound = 0;
upper_bound = 3; 
while (upper_bound - lower_bound) > bisection_precision
    beta = (lower_bound + upper_bound) / 2;
    [s1,c1] = polynomial(x,deg_s1);
    F = [sos(s1), sos(-(beta - h)*s1 - (V-gamma))];
    sol = solvesos(F,[],ops,c1);
    if sol.problem == 0
        lower_bound = beta;
    else
        upper_bound = beta;
    end
end
beta_val = lower_bound

for i=1:100
    % Solve for s1 and s2
    fprintf("Iteration %d: ",i);
    [s1,c1] = polynomial(x,deg_s1);
    [s2,c2] = polynomial(x,deg_s2);
    F = [sos(s1), sos(s2), sos(-(beta_val - h)*s1 - (V-gamma)), sos(-(l + dVdt) + s2*(V - gamma))];
    sol = solvesos(F,[],ops,[c1;c2]);

    if sol.problem ~= 0
        %error('The problem is infeasible!');
        warning(['Searching for s1 and s2 is infeasible after ', num2str(i), ' iterations! Using the last feasible solution.']);
        break;
    end

    % Update s1, s2
    s1 = replace(s1, c1, value(c1));
    %sdisplay(s1);
    s2 = replace(s2, c2, value(c2));
    %sdisplay(s2);

     % Solve for V and beta
    [V,cV] = polynomial(x,deg_V);  
    dVdt = jacobian(V,x)*f;
    beta = sdpvar(1);

    F = [sos(V-l), cV(1)==0, sos(s1), sos(-(beta - h)*s1 - (V-gamma)), ...
         sos(-(l + dVdt) + s2*(V - gamma))];
    sol = solvesos(F,-beta,ops,[cV; beta]);

    if sol.problem ~= 0
        %error('The problem is infeasible!');
        warning(['Searching for V and beta is infeasible after ', num2str(i), ' iterations! Using the last feasible solution.']);
        V_infeasible = 1;
        break;
    end

    V = replace(V, cV, value(cV));
    %sdisplay(V);
    dVdt = jacobian(V,x)*f;

    % Save last feasible solution
    V_last_feasible = V;
    beta_last_feasible = value(beta);

    % Check the stopping criterion
    if abs(value(beta) - beta_val) < stop_precision
        disp(['Stopping criterion reached after ', num2str(i), ' iterations. Exiting the loop.']);
        break;
    end
    beta_val = value(beta)

    %% update h
%     [Coeffs, Terms] = coefficients(V, [x1, x2]);
%     
%     quadratic_term = 0;
%     for i = 1:length(Terms)
%         powers = degree(Terms(i), [x1, x2]);
%         if sum(powers) == 2 && max(powers) <= 2
%             quadratic_term = quadratic_term + Coeffs(i) * Terms(i);
%         end
%     end
%     
%     h = quadratic_term;

end

elapsed_time = toc;

% If the last iteration was infeasible for V, use the last feasible solution
if V_infeasible == 1
    V = V_last_feasible;
    beta_val = beta_last_feasible;
end

% Convert f, V and beta to strings
f_string = sdisplay(f);
V_string = sdisplay(V);

% Print the final V and beta to a text file
filename = strcat('output_', name, '.txt');
fileID = fopen(filename,'w');
fprintf(fileID, 'f:\n');
fprintf(fileID, '[%s]\n\n', strjoin(f_string, '; '));
fprintf(fileID, 'Final V:\n');
fprintf(fileID, '%s\n\n', V_string{1});
fprintf(fileID, 'Final beta: %.4f\n', beta_val);
fprintf(fileID, 'Iteration Number: %d\n', i);
fprintf(fileID, 'Execution Time: %.2f seconds\n', elapsed_time);
fclose(fileID);