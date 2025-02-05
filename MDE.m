function [finalY, finalbeta, finalbeta0, y, finalinfo, allY, value] = MAIN(varargin)

% Input:
%   1: beta grid
%   2 to N+1: contact maps
%   
% Output:
%   finalY: final N * 3 structure
%   finalbeta: selected beta
%   y: log-likelihood, first row is EDM constrained, second row is
%       un-constrained
%   finalinfo: EDM embedding info of the selected beta
%   allY: all structures under beta grid
%   value: stress values under beta grid

betavec = varargin{1};
N = (numel(varargin) - 1);
Nbeta = size(betavec); Nbeta = Nbeta(2);
sizeN = size(varargin{2}); sizeN = sizeN(1);
contactMap = zeros(sizeN, sizeN, N);
vecN = [];
for i = 1:N
    % first make sure contact map is symmetric
    contactMap(:, :, i) = (varargin{i + 1} + transpose(varargin{i + 1})) ./ 2;
    tmp = contactMap(:, :, i);
    % Convert contact map read count to vector
    vecN = vertcat(vecN, tmp(tril(true(size(tmp)), -1)));
end

% Preapre EMBED.m parameters
pars.plotyes = 0;
pars.m = 0;
pars.anchorconstraintyes = 0;
pars.spathyes = 0;
pars.refinement = 0;
% pars.pseudocount = 1;
dim = 3;

% inverse averaged contact
data = mean(contactMap, 3);
data = 1 ./ (data + 1);

% Initialization; choose beta = 1
beta = 1;
Dtemp = data .^ (2. / beta);% updated averaged dissimilarity matrix

% fit structure
[Y, infos] = EMBED(Dtemp, dim, pars);
finalY = transpose(Y);
finalinfo = infos;
Dpredict = transpose(pdist(finalY)); %obtain distance matrix
[finallikelihood, finalbeta0] = PoiLikelihood(vecN, vertcat(Dpredict .^ 2, Dpredict .^ 2), -beta / 2.);
finalbeta = beta;
finalY = LinearFiltering(finalY);

% Grid search for optimal beta
y = zeros(2, Nbeta);
value = zeros(1, Nbeta);
i = 1;
allY = zeros(sizeN, 3, N);
for beta = betavec
    Dtemp = data .^ (2 / beta);
    [Y, infos] = EMBED(Dtemp, dim, pars); % EDM embedding
    value(1, i) = infos.f; % stress value
    Y = transpose(Y);
    Dpredict = transpose(pdist(Y)); % distance vector
    [likelihood, beta0] = PoiLikelihood(vecN, vertcat(Dpredict .^ 2, Dpredict .^ 2), -beta / 2.); % evaluate likelihood
    allY(:, :, i) = LinearFiltering(Y); % linear filtering
    y(1, i) = likelihood;
    y(2, i) = PoiLikelihood(vecN, vertcat(Dtemp(tril(true(size(Dtemp)), -1)), Dtemp(tril(true(size(Dtemp)), -1))), -beta / 2.); % unconstrained likelihood
    if (likelihood > finallikelihood)
        finallikelihood = likelihood;
        finalbeta = beta;
        finalbeta0 = beta0;
        finalY = allY(:, :, i);
        finalinfo = infos;
    end
    i = i + 1;
end
return;
