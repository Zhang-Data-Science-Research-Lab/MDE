function [Y, infos] = rHENewton2_beta(D, pars)
%% This code is designed to find %%%%%%%%%%%%%%%%%%%%%
%% This code is designed to find %%%%%%%%%%%%%%%%%%%%%
% Nearest Euclidean Distance Matrix with Imbedding Dimension <= r
%
% min 0.5\| H o (Y-D) \|^2
% s.t. diag(Y) = 0, (-Y) \in \K^n_+(r)
% extra fixed constraints: Y_{ij} = D_{ij} for (i, j) \in (I, J)

%%
% Input: D   -- Predistance matrix or Dissimilarities (diag(D)=0, D >= 0)
% 
% pars
%  pars.H -- n-by-n H weight matrix; if H = w is a vector, it is diagonal
%            \| H o(Y-D)\|^2 or \| W^{1/2} (Y-D) W^{1/2} \|^2
%  pars.b -- the right-hand side vector, containing fixed distances
%  pars.I -- 
%  pars.J -- D(i, j) = bij fixed distances
%           =[] if no fixed distances are present
%           when b \not = [], equations are arranged as
%           [diag(Y); Yij] = [0; b]
%  pars.eig -- 1 use eig
%              0 use mexeig
%  pars.r  -- Imbedding dimension (default r = n, full dimension)
%  pars.Y0 -- Initial point 
%               Default: PCA (Principal Coordinator Analysis)
%               modification on NEDM by ENewton.m
%  pars.tolrank -- error tolerance for the matrix rank (1.0e-8, default)
%  pars.tolrel -- relative error tolerance (1.0e-5 default)
%                relative error is defined to be
%                (sqrt(f0) - sqrt(f))/(sqert(f0)*n^2) 
%                (average relattive error per element) 
% pars.spathyes = 1 -- the shortest paths to replace 
%                 the missing (0) distances  (default)
%                 0 -- the shortest paths not used
% PLOT parameters
% pars.plotvarianceyes = 1 -- plot fall-off of residual variance with dims
%                       default: no plot
% pars.plot2dimyes = 1 -- plot the leading 2 dimensions X'(:,1), X'(:,2)
%                    0 -- no plot
% pars.plot3dimyes = 1 -- plot the leading 2 dimensions 
%                         X'(:,1), X'(:,2), and X'(:, 3)
%                    0 -- no plot
%% Output:
%        Y  -- EDM with imbedding dimension <= r
% INFOS
%   INFOS.X  -- Final configuration of n points X=[x1, x2, ..., xn]
%                                                 xi \in \Re^r  
%
%% Comments:
% Implementation framework follows the pattern of PenCorr of Defeng Sun
% 
%
%            1st  version: July 3,  2012
%            This version: April 9, 2013
%
%% Step 0: Set up parameters and 
%          Check whether H=E or
%          Check whether H=w, diagonally weighted
%
t0   = tic;
n    = length(D);
Dold = D;

if ~isfield(pars, 'r')
    pars.r = n;
end
r = pars.r;

% to use the shortest paths for missing distances
if ~isfield(pars, 'spathyes')
    pars.spathyes = 1;
end
spathyes = pars.spathyes;

if (spathyes) && ((n^2 - nnz(D)) > n)
    D = sqrt(D);
    fprintf('\n rHENewton2: Comupting the shortest paths ...');
    D = graphallshortestpaths(D);
    D = max(D.^2, Dold); % to only replace those missing distances
end

Hcase = 1; % to indicate H is n-by-n weight matrix
if isfield(pars, 'H')
    H = pars.H;
    diagweight = size(H, 2);
else
    H = [];
end

if (isempty(H)) % = 1 means H = E
    fprintf('\n Uniform Weight');
    pars.printyes = 1;
    [Y, infos] = rENewton2_beta(D, pars);
    if spathyes
        infos.f = 0.5*sum(sum((Y-Dold).^2));
    end
    Hcase = 0;
    diagweight = 0;
end

if (diagweight == 1) % H is a vector, diagonal weight
    w = H;
    fprintf('\n Diagonal Weight');
    pars.printyes = 1;
    [Y, infos] = rDENewton2_beta(D, w, pars);
    if spathyes
        Dold = (sqrt(w)*sqrt(w)').*(Y-Dold);      
        infos.f = 0.5*sum(sum(Dold.^2));
    end
    Hcase = 0;
end
%%%%%%% End of Step 1 checking

%% Step 2: (Otherwise),  H is n-by-n weight matrix
%          calculate the weight vector and read more information

%%%%%%%%%% Start of Hcase
if Hcase
    if ~isfield(pars, 'tau')
       pars.tau = 0.1;
    end
    tau = pars.tau;
    H =0.5*(H+H');
    hsum = sum(H, 2);
    if (any(hsum == 0))
        error('The graph of the data is not connected!');
    end
        
    w = max(H)';
    w = max(tau, w);
    invw = 1./w;
    H2 = H.*H;
    NoOfNonzeros = nnz(H)/2;
    equalweightflag = any(w/max(w) - ones(n,1));
if (equalweightflag == 0)
    fprintf('\n Subproblems all have equal diagonal weights');
end

% Read more information
if ~isfield(pars, 'eig')
    pars.eig = 1; % use eig
end

if ~isfield(pars, 'tolrel')
    pars.tolrel = 1.0e-5;  % 1.0e-4 (another choice)
end
tolrel = pars.tolrel;

if ~isfield(pars, 'tolrank')
    pars.tolrank = 1.0e-8;
end
tolrank = pars.tolrank;

if~isfield(pars, 'EigenRatioLevel')
    pars.EigenRatioLevel = 90/100;
end
EigenRatioLevel = pars.EigenRatioLevel;

% set other tolerance based on given
innerTolrel = tolrel;
tolsub      = max(innerTolrel, 1.0*tolrel);  
tolPCA      = max(innerTolrel, 1.0*tolrel);
tolKKT      = 1.0e-5;
%tolinfeas   = max(innerTolrel, 1.0*tolrel);   %% the feasibility tolerance 
tolsub_rank = tolsub;  

% other parameters
if ~isfield(pars, 'maxit')
    pars.maxit = 200;
end
maxit    = pars.maxit; 
maxitsub = 100;
%
finalPCA = 0;
residue_cutoff = 0;
% residue_cutoff = 10;
% if tolrel <= 1.0e-4;
%    residue_cutoff = 100;
% end

% constant pars
const.disp1 = 10;
const.disp2 = 10;
const.rank_hist    = 5;  const.rank_hist    = max(2, const.rank_hist);  
const.rankErr_hist = 3;  const.rankErr_hist = max(2, const.rankErr_hist);  
const.funcVal_hist = 2;  const.funcVal_hist = max(2, const.funcVal_hist);
const.residue_hist = const.funcVal_hist;  
const.residue_hist = max(2, const.residue_hist); 
const.lancelot_hist = 3;
rank_hist    = zeros(const.rank_hist,1);
rankErr_hist = zeros(const.rankErr_hist,1);
funcVal_hist = zeros(const.funcVal_hist,1);
residue_hist = zeros(const.residue_hist,1);
progress_rank1    = 1;
progress_rank2    = 1;
%progress_residue  = 1.0e-4;
progress_relErr   = tolrel;
progress_rankErr  = 1.0e-3;

% penalty pars
c0_min = 1.0;
c0_max = 1e3;   
alpha_min = 1.2; %1.4
alpha_max = 4.0;
c_max  = 1.0e8;

% counters
Totalcall_EN    = 0;
Totaliter_EN    = 0;
TotalEigenD     = 0;

% output problem information
% check whether there are fixed distances
fixdistance = 0;
if ~isfield(pars, 'b')
    n0 = 0;
    k = n;
else
    b = pars.b;
    I = pars.I;
    J = pars.J;
    n0 = length(b);
    if n0 > 1
        fixdistance = 1;
    end
    k = n+n0;
end

fprintf('\n The problem information: \n');
fprintf(' Dimension of SDP constr.           = %3.0f \n', n);
fprintf(' Number of fixed distances          = %3.0f \n', n0);
fprintf(' The required embedding dimension  <= %3.0f \n', r);

%% Step 3: Get started for the H case
%

fprintf('\n ^^^^^^^^ Computing Initial Point by DENewton  ^^^^^^^^ ')

pars.computingXyes = 1;

y = zeros(k, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if n0 <= 29*15 % 30 anchors
%   [Y, infos] = DENewton2_beta(D, w, pars);
%   y      = infos.y;
% else
%     [Y, infos] = DENewton_beta(D, w, pars); % without any finxed constraints
%     y(1:n) = infos.y;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Y, infos] = DENewton_beta(D, w, pars); % without any finxed constraints
y(1:n) = infos.y;

% make use of information provided by DENewton2_beta
pars.y = y;            % initial Lagrangian multiplier
P      = infos.P;      % eigen information is on (-0.5*JYJ) in ENewton.m
                       % Y is the computed EDM
                       % P are (rank_Y) leading eigenvectors
lambda = infos.lambda; % positive eigenvalues of (-0.5*JYJ) in non-increasing order
                       % its length is infos.rank
rank_Y = infos.rank;   % length(lambda) = rank_Y;

%%
Totalcall_EN   = Totalcall_EN + 1;
Totaliter_EN   = Totaliter_EN + infos.Iter;
TotalEigenD    = TotalEigenD  + infos.EigenD;

residue_ENewton = 0.5*sum(sum( ( H.*(Y-D)).^2 )); % 0.5\| H o(Y-D)\|^2
infos.f         = residue_ENewton;
residue_ENewton = sqrt(2*residue_ENewton); % \| Y-D\|

% Decide whether the obtained solution by ENewton is already good enough
if rank_Y <= r
    rankErr_ENewton = 0;
    EigenRatio = 1;
else
    rankErr_ENewton = abs( sum(lambda(1:rank_Y)) - sum(lambda(1:r)) );
    EigenRatio = sum(lambda(1:r))/sum(lambda(1:rank_Y));
end

%if  ( rankErr_ENewton <= tolrank ) 
    % Check if KKT condition is satisfied
if fixdistance
   KKT = sparse(I, J, y(n+1:end), n, n);
   KKT = 0.5*(KKT+KKT') + diag(y(1:n));
else
   KKT = diag(y);
end
KKT = H2*(Y-Dold) + KKT - infos.Z;
Err_KKT = sum(sum(KKT.^2))^0.5;

if ( (EigenRatio >= EigenRatioLevel) && (equalweightflag == 0) && (fixdistance == 0) ) || ( Err_KKT <= tolKKT )
%if ((rank_Y <= r) && (equalweightflag == 0) && (fixdistance == 0) ) || ( Err_KKT <= tolKKT )
   fprintf('\n The initial point by DENewton2 is a good approximation!')
   fprintf('\n The Embedding dimension of NEDM  = %2.0d', rank_Y)
   fprintf('\n The rank error                   = %4.3e', full(rankErr_ENewton))
   fprintf('\n Residue_ENewton                  = %9.8e', full(residue_ENewton))
   fprintf('\n EigenRatio                       = %3.2e', full(EigenRatio))
   time_used = toc(t0);
   fprintf('\n Total computing time             = %.1f(secs) \n', time_used); 
   infos.Iter    = 0;
   infos.callEN  = Totalcall_EN;
   infos.itEN    = Totaliter_EN;
   infos.EigenD  = TotalEigenD;
   infos.rank    = rank_Y;
   infos.rankErr = rankErr_ENewton;
   infos.residue = residue_ENewton;
   infos.EigenRatio = EigenRatio;
   
   if rank_Y < r
       fprintf('\n The embedding dimension of the initial EDM is < %2.d', r);
       X = infos.X;

       X = [X; zeros((r-rank_Y), n)];
       infos.X = X;
   end
       %infos.f      has been assigned
       %infos.X      embedding points are already obtained by ENewton
   infos.t       = time_used;
   return;
end
%end

% Otherwise, it is not good enough. Output some key information
fprintf('\n The Embedding dimension of NEDM           = %2.0d', rank_Y)
fprintf('\n The rank error of NEDM                    = %4.3e', full(rankErr_ENewton))  
fprintf('\n Residue_ENewton                           = %9.8e', full(residue_ENewton))
fprintf('\n Eigen Ratio: sum(lambda(1:r))/sum(lambda) = %3.2e', full(EigenRatio))
residue_1 = residue_ENewton;

%%%%%% We choose not to calculate the cMDS, as it is unlikely to succeed
% %% Calculate the classical MDS (cMDS) based on Y (just obtained by ENewton)
% % and test whether it is good enough
% %
% lambda1 = max(lambda(1:r), 0).^0.5;
% X = P(:, 1:r)*diag(lambda1);
% X = X';
% Y1 = sqdistance(X);
% %%%%%%%%%%%%%%%%%%%%%%%%% end of cMDS%
% 
% residue_ENewtonMDS = sum(sum(( (Y1-D).*(sqrtw*sqrtw') ).^2));
% residue_ENewtonMDS = residue_ENewtonMDS^0.5; 
% residue_error = abs( residue_ENewtonMDS - residue_ENewton );
% fprintf('\n Residue_ENewtonMDS  = %9.8e',full(residue_ENewtonMDS))
% %
% if ( residue_error/max(residue_cutoff, residue_ENewtonMDS ) ) <= tolPCA 
%     fprintf('\n ENewton_cMDS is good enough!')
%     time_used = toc(t0);
%     fprintf('\n Total computing time   = %.1f', time_used);
%     infos.Iter    = 0;
%     infos.callEN  = Totalcall_EN;
%     infos.itEN    = Totaliter_EN;
%     %INFOS.itBiCG  = Totalnumb_BiCG;
%     infos.EigenD  = TotalEigenD;
%     infos.rank    = r;
%     infos.rankErr = 0;
%     infos.residue = residue_ENewtonMDS;
%     infos.f       = 0.5*residue_ENewtonMDS^2;
%     infos.X       = X;                       % Embedding points in cloumns
%     Y             = Y1;                      % Output cMDS as Y
%     infos.P       = P(1, 1:r);
%     infos.lambda  = lambda(1:r);
%     infos.t       = time_used;
%     return;
% end
% %
% % Otherwise, cMDS is not good enough. It may be used as initial point
% % We simply use Y by ENewton as initial point
% % We may also follow Defeng to use alternating projection method to
% % generate an initial matrix Y
%%%%%%% enf of cMDS

%% Step 4: Solving the Majorized Subproblems
% initialize U
%
P1 = P(:,1:r);            %U  = P1*P1', % no need to form U to save cost
lambda = max(0, lambda);
qY = sum(lambda(1:rank_Y)) - sum(lambda(1:r));
rankErr = abs(qY);

% initial penalty parameter c
%c0 = 0.50*(residue_ENewtonMDS^2 - residue_ENewton^2);
% c0 = c0_min;
% c0 = 0.25*c0/max(1.0, rankErr_ENewton); 

if rankErr_ENewton >= 1.0e3
    c0 = sqrt(rankErr_ENewton);
elseif rankErr_ENewton >= 10
    c0 = rankErr_ENewton;
else
    rankErr_ENewton = max(rankErr_ENewton, Err_KKT);
    c0 = sqrt(rankErr_ENewton);
    c0 = max(10, c0);
    %c0 = max(10, rankErr_ENewton);
end

c0 = min(200, c0);
% if Err_KKT >= 1.0e3
%     c0 = 10*sqrt(Err_KKT);
% else
%     c0 = max(1, Err_KKT);
% end

if r <= 1
    c0 = c0_max;
end
%
if tolrel >= 1.0e-1;  %% less acurate, larger c
    c0 = 4*c0;
elseif tolrel > 1.0e-2;  %% less acurate, larger c
    c0 = 2*c0;
% elseif tolrel >= 1.0e-3;  %% less acurate, larger c
%     c0 = 2*c0;
% elseif tolrel >= 1.0e-4;  %% less acurate, larger c
%     c0 = 2*c0;
end
% if tolrel <= 1.0e-6;  %% more acurate, smaller c
%     c0 = c0/2;
% end
c0 = max(c0, c0_min);
c0 = min(c0, c0_max);
c  = c0;
%
fprintf('\n\n ******************************************************* \n')
fprintf( '      Sequential Semismooth Newton Method Starts!!!      ')
fprintf('\n   ******************************************************* \n')
fprintf('The initial rank        = %3.0d \n', rank_Y);
fprintf('The initial rank error  = %4.3e \n', full(rankErr));
fprintf('The initial ||Y0-D||    = %5.4e \n', full(residue_1));
%
relErr_0    = 1.0e6; % set initial relative error large enough
break_level = 0;
%
k1        = 1; % count of outer iteration: initial value 1 because of cMDS
sum_iter  = 0; 
residue_hist(1) = residue_1;
while ( k1 <= maxit )    
    subtotaliter_EN    = 0;
    subtotalEigenD     = 0;
    
    fc = 0.5*residue_1^2;
    fc = fc + c*qY;         
    tt = toc(t0);
    fprintf('\n ============')
    fprintf(' The %2.0dth level of penalty par. c = %3.2e',k1, full(c))
    fprintf('  ====================')
    fprintf('\n ........Calling DENewton')
    fprintf('\n CallNo.  NumIt    RankY    EigenRatio   RankErr      Sqrt(2*FunVal)     Time');   
    fprintf('\n %2.0fth  %4.0s     %5.0d   %3.2   %4.3e        %8.4e      %.1f', ...
        0,  '--',   rank_Y,    full(EigenRatio),  full(rankErr),   full(sqrt(2)*fc^0.5),   tt);  
%
% solve the subproblem with fixed c
%
    rank_hist(1) = rank_Y;
    funcVal_hist(1) = fc^0.5;
%
% creating the matrix Dk for the subproblem
% Dk = D + c*(J(I-U)J)
%    = D - c/n - cJUJ (ignore the diagonal part c*I)
%
Dk = Y - (invw*invw').*H2.*(Y-D);
P1 = sparse(diag(invw))*P1 - (1/n)*(invw*sum(P1));
Dk = Dk - c*((invw*invw')/n + (P1*P1'));
%Dk(1:(n+1):end) = Dk(1:(n+1):end)+ c;
Dk(1:(n+1):end) = 0;

    for itersub = 1:maxitsub 
        %pars.printyes = 1;
        [Y, infos] = DENewton2_beta(Dk, w, pars); % use pars.y to warm start
        
        P          = infos.P; % eigen information is for (-0.5*JYJ)
        lambda     = infos.lambda;
        rank_Y     = infos.rank;
        pars.y     = infos.y;
        
        Totalcall_EN         = Totalcall_EN + 1;
        subtotaliter_EN      = subtotaliter_EN + infos.Iter;
        subtotalEigenD       = subtotalEigenD  + infos.EigenD;
        
        lambda = max(lambda, 0);
        if rank_Y <= r
            qY = 0;
        else
            qY = sum(lambda(1:rank_Y)) - sum(lambda(1:r));
        end
        rankErr = qY;
        
        EigenRatio = sum(lambda(1:r))/sum(lambda(1:rank_Y));
        fc         = sum(sum( (H.*(Y-D)).^2 ));
        residue_1  = fc^0.5;        
        fc         = 0.5*fc + c*qY;  
        if ( itersub <= const.disp1 || mod(itersub,const.disp2) == 0 )
            dispsub = 1;
            tt = toc(t0);
           fprintf('\n %2.0fth  %4.0f     %5.0d        %3.2e   %4.3e   %8.4e        %.1f',...
                itersub, infos.Iter, rank_Y, full(EigenRatio), full(rankErr), full(sqrt(2)*fc^0.5), tt)
        else  
            dispsub = 0;
        end
        %     
        %%% record rank history
        %
        if  itersub <= const.rank_hist - 1
            rank_hist(itersub + 1) = rank_Y;
        else
            for j = 1:const.rank_hist-1
                rank_hist(j) = rank_hist(j+1);
            end
            rank_hist(const.rank_hist) = rank_Y;
        end
        %
        %%% record function value history
        %
        if  itersub <= const.funcVal_hist - 1
            funcVal_hist(itersub + 1) = sqrt(fc);
        else
            for j = 1:const.funcVal_hist-1
                funcVal_hist(j) = funcVal_hist(j+1);
            end
            funcVal_hist(const.funcVal_hist) = sqrt(fc);
        end
        %
        %%% record residue history
        %
        if sum_iter + itersub <= const.residue_hist -1
            residue_hist(sum_iter + itersub + 1) = residue_1;
        else
            for j = 1:const.residue_hist-1
                residue_hist(j) = residue_hist(j+1);
            end
            residue_hist(const.residue_hist) = residue_1;
        end
        %
        if rankErr <= tolrank
            tolsub_check = tolsub_rank;       
        else
            tolsub_check = tolsub*max(10, min(100,rank_Y/r));
        end
        %
        if itersub >= const.funcVal_hist - 1
            relErr_sub  = abs(funcVal_hist(1) - funcVal_hist(const.funcVal_hist));
            relErr_sub  = relErr_sub/max(residue_cutoff, ...
             max(abs(funcVal_hist(1)), abs(funcVal_hist(const.funcVal_hist))) );
            relErr_sub  = sqrt(2)*relErr_sub/n^2;
        end
        %
        %%% when to exit the subproblem
        %
        if ( itersub >= const.funcVal_hist - 1 &&  relErr_sub <= tolsub_check )
            if dispsub == 0 % display the exit information if not yet done
                tt = toc(t0);
                fprintf('\n %2.0fth  %4.0f     %5.0d        %3.2e   %4.3e   %8.4e        %.1f',...
                itersub, infos.Iter, rank_Y, full(EigenRatio), full(rankErr), full(sqrt(2)*fc^0.5), tt)
            end
            break;
        elseif  ( itersub >= const.rank_hist-1 && abs( rank_hist(1) - rank_hist(const.rank_hist) ) <= progress_rank1...
                && rank_Y - r >= progress_rank2 )
            if dispsub == 0
                tt = toc(t0);
               fprintf('\n %2.0fth  %4.0f     %5.0d        %3.2e   %4.3e   %8.4e        %.1f',...
                itersub, infos.Iter, rank_Y, full(EigenRatio), full(rankErr), full(sqrt(2)*fc^0.5), tt)                
            end
            break;
        end
%
% end of exit of subproblem and 
% update U, G0
%
        P1  = P(:, 1:r);          %U   = P1*P1';
        Dk = Y - (invw*invw').*H2.*(Y-D);
        P1 = sparse(diag(invw))*P1 - (1/n)*(invw*sum(P1));
        Dk = Dk - c*((invw*invw')/n + (P1*P1'));
        %Dk(1:(n+1):end) = Dk(1:(n+1):end)+ c;
        Dk(1:(n+1):end) = 0;
    end   % end of subproblem
    %
    sum_iter        = sum_iter + itersub;  
    Totaliter_EN    = Totaliter_EN + subtotaliter_EN;
    TotalEigenD     = TotalEigenD  + subtotalEigenD;
    fprintf('\n SubTotal %2.0f       %2.0f (EigenD)',...
        subtotaliter_EN, subtotalEigenD);
   %
    if sum_iter >= const.residue_hist-1
        relErr = abs(residue_hist(1) - residue_hist(const.residue_hist));
        relErr = relErr/max(residue_cutoff, max(residue_hist(1), residue_hist(const.residue_hist)));
        relErr = relErr/NoOfNonzeros; % averaging relative error
    else
        relErr = abs(residue_hist(1) - residue_hist(sum_iter));
        relErr = relErr/max(residue_cutoff, max(residue_hist(1), residue_hist(sum_iter)));
        relErr = relErr/NoOfNonzeros; % averaging relative error
    end
    tt = toc(t0);
    [hh,mm,ss] = time(tt);
    fprintf('\n   Iter.  PenPar.     Rank(Y)     EigenRatio     RankError     RelErr     ||Y-D||      Time_used ')
    fprintf('\n   %2.0d     %3.2e    %3.0d        %3.2e      %5.4e   %3.2e  %5.4e     %d:%d:%d \n',...
        k1, full(c), rank_Y, full(EigenRatio),  full(rankErr), full(relErr), full(residue_1), hh,mm,ss)
%
    if  k1 <= const.rankErr_hist
        rankErr_hist(k1) = rankErr;
    else
        for j=1:const.rankErr_hist-1
            rankErr_hist(j) = rankErr_hist(j+1);
        end
        rankErr_hist(const.rankErr_hist) = rankErr;
    end    
    %%% termination test
    if ( relErr <= tolrel )
        %if ( rankErr <= tolrank )
        if ( EigenRatio >= EigenRatioLevel ) || ( rankErr <= tolrank )
            fprintf('\n The rank constraint is satisfied!')
            break;
        elseif ( k1 >= const.rankErr_hist && abs(rankErr_hist(1) - rankErr_hist(const.rankErr_hist)) <= progress_rankErr )
            fprintf('\n Warning: The rank does not decrease any more! :( ')
            finalPCA = 1;
            break;
        end
    else
        if ( abs(relErr_0 - relErr)/max(1,relErr) <= progress_relErr )
            break_level = break_level + 1;
            if break_level == 3
                fprintf('\n Warning: The relErr is consecutively decreasing slowly, quit! :( ')
                if ( rankErr > tolrank )
                    finalPCA = 1;
                end
                break;
            end
        end        
    end
    %
    k1        = k1 + 1;
    relErr_0  = relErr;
    
    %%% update c
    if rank_Y <= r
        c = min(c_max, c);
        fprintf('\n The rank constraint is satisfied and keep c the same!')
    else
        if rankErr/max(1, r) > 1.0e-1
            c = min(c_max, c*alpha_max);
        else
            c = min(c_max, c*alpha_min);
        end
    end    
end
residue_1 = sum(sum( ( (Y-Dold).*H ).^2 ))^0.5;

%% Step 5: Output (final tidy up)
% return to Dold

%% make sure the embedding points are in Re^r
if rank_Y > r     % th excessive rank beyond r to be ignored
   X = infos.X;
   infos.X = X(1:r, :);
   infos.lambda = lambda(1:r);
   Y       = sqdistance(X);
   rank_Y  = r;
   rankErr = 0;
   residue_1 = sum(sum( ( (Y-Dold).*H ).^2 ))^0.5;
end
%%
% Final cMDS correction (if required) and Final configuration of n points in Re^r
%
if finalPCA
  fprintf('\n Final cMDS is needed!')  
  infos.lambda = lambda(1:r);
  infos.P = P(:, 1:r);
  lambda1 = abs(lambda(1:r)).^0.5;
  X = P(:, 1:r)*diag(lambda1);
  X = X';
  infos.X = X;
  Y       = sqdistance(X);
  rank_Y  = r;
  rankErr = 0;
  residue_1 = sum(sum( ( (Y-Dold).*H ).^2 ))^0.5;
end

infos.Iter      = k1;
infos.callEN    = Totalcall_EN;
infos.itEN      = Totaliter_EN;
infos.EigenD    = TotalEigenD;
infos.rank      = rank_Y;
infos.rankErr   = rankErr;
infos.resErr    = relErr;
infos.residue   = residue_1;
infos.f         = 0.5*residue_1^2;
%INFOS.infeas    = NormInf;

end
%%%%%% End of Hcase
time_used = toc(t0);
infos.t   = time_used;
fprintf('\n Primal function value 0.5|| Ho(Y-D)||^2    === %9.8e', full(infos.f));
%fprintf('\n MajorDual function value === %9.8e', major_dualVal);
fprintf('\n Computing time           ======= %.1f(secs) \n', infos.t);
%%%%
%%%%%%%%% Enf of Output

%% Step 6: Graphics
%%% plot 2-dimesional embedding

X = infos.X;
X = X';

if ~isfield(pars, 'plot2dimyes')
    pars.plot2dimyes = 0; % no plot
end
plot2dimyes = pars.plot2dimyes;

if plot2dimyes && (r>=2)
    figure
    plot(X(:, 1), X(:,2), 'ro');
    title('Two-dimensional Emap');
end

%%% plot 3-dimesional embedding
if ~isfield(pars, 'plot3dimyes')
    pars.plot3dimyes = 0; % no plot
end
plot3dimyes = pars.plot3dimyes;

if plot3dimyes && (r>=3)
    figure
    plot3(X(:, 1), X(:,2), X(:, 3), 'bd');
    title('Three-dimensional Emap');
end
%%%%%%%%% END of Graphics

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% end of the main program %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%%  **************************************
%%  ******** All Sub-routines  ***********
%%  **************************************
%%% To change the format of time 
function [h,m,s] = time(t)
t = round(t); 
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
return
%%% End of time.m

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function X = cMDS(P, lambda, Rank)
% %lambda>=0 and b>=0
% n = length(lambda);
% if Rank>0
%     P1       = P(:, 1:Rank);
%     lambda1  = lambda(1:Rank);
%     %lambda1  = lambda1.^0.5;
%     lambda1  = lambda1.^0.5/sqrt(2); % P1*P1' = 0.5 (JYJ)
%     if Rank>1
%         P1 = P1*sparse(diag(lambda1));
%     else
%         P1 = P1*lambda1;
%     end
%    P1 = P1'; %column vectors used to calculate the distances
%    X  = sqdistance(P1);
% else
%     X = zeros(n,n);
% end
% return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%