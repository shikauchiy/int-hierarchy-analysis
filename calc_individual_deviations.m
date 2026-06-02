function [metrics,corrRes,metricTable] = calc_individual_deviations(INT,isTDC,AASP,covar,covarNames)
% [metrics,corrRes,metricTable] = ...
%   calc_individual_deviations(INT,isASD,AASP)
%% ==============================================================
% Computes individual deviations from a TDC-derived INT hierarchy
% template and examines their associations with sensory traits
% measured by the AASP.
%
% Individual INT profiles were modeled as:
%
%   INT_subject = alpha + beta * INT_template + residual
%
% where:
%   alpha = global shift
%   beta  = hierarchical scaling
%   RMS   = sqrt(mean(residual.^2))
%
% Input:
%   INT        : [Nsub x 360] intrinsic neural timescales
%                for each participant (Glasser 360 parcels)
%   isTDC      : [Nsub x 1] logical vector indicating TDC participants
%   AASP       : [Nsub x 4] AASP scores or principal component scores
%
% Optional:
%   covar      : [Nsub x K] covariates (e.g., Age, Sex, FD)
%   covarNames : names of covariates
%
% Output:
%   metrics    : structure containing
%                alpha (global offset),
%                beta (hierarchical scaling),
%                rms (residual deviation)
%   corrRes    : table of correlations between metrics and AASP scores
% ==============================================================
%% Parameter settings
aaspNames = {'AASP_PC1','AASP_PC2','AASP_PC3','AASP_PC4'};
metricNames = {'alpha','beta','rms'};
%% Template stability analysis
% Evaluate whether exclusion of a single TDC participant
% changes the parcel ranking of the TDC template.
[N,P] = size(INT);

% Leave-one-out template construction for TDC participants.
% For ASD participants, the full TDC template is used.
template_full = mean(INT(isTDC,:),1,'omitnan');
[~, sort_full] = sort(template_full,'ascend');

tdcIdx = find(isTDC);
nTDC = numel(tdcIdx);

rank_shift = zeros(nTDC,1);        % maximum rank shift
n_swapped  = zeros(nTDC,1);        % number of parcels with rank changes

for k = 1:nTDC
    i = tdcIdx(k);

    % Sort parcels according to the subject-specific template hierarchy.
    % Individual INT values are reordered using the same ranking.
    template_loo = mean(INT(isTDC & ( (1:size(INT,1))'~=i ),:),1,'omitnan');
    [~, sort_loo] = sort(template_loo,'ascend');

    rank_full = zeros(P,1); rank_full(sort_full) = 1:P;
    rank_loo  = zeros(P,1); rank_loo(sort_loo)  = 1:P;

    diff_rank = abs(rank_full - rank_loo);

    rank_shift(k) = max(diff_rank);
    n_swapped(k)  = nnz(diff_rank>0);
end

fprintf('Max rank shift (median): %.1f\n', median(rank_shift));
fprintf('Parcels swapped (median): %.0f / %d\n', median(n_swapped), P);
%% Input validation
assert(P==360, 'INT must be [Nsub x 360].');
assert(size(AASP,1)==size(INT,1) && size(AASP,2)==4, 'AASP must be [Nsub x 4].');
assert(numel(isTDC)==size(INT,1), 'isTDC must be [Nsub x 1].');

if ~exist('covar','var') || isempty(covar)
    covar = [];
    covarNames = {};
else
    assert(size(covar,1)==N, 'covar must have N rows.');
    if ~exist('covarNames','var') || isempty(covarNames)
        covarNames = arrayfun(@(k)sprintf('covar%d',k), 1:size(covar,2), 'uni', 0);
    end
end

metrics = struct();
metrics.alpha    = nan(N,1);
metrics.beta     = nan(N,1);
metrics.rms      = nan(N,1);

tdcIdx = find(isTDC);
tdcSum = sum(INT(isTDC,:), 1, 'omitnan');  % [1 x 360]
tdcN   = numel(tdcIdx);
template_full = tdcSum / tdcN;
%% Template fitting
% INT_subject = alpha + beta * template + residual
%
% alpha : global offset from the template
% beta  : hierarchical scaling relative to the template
% rms   : magnitude of residual deviation after accounting
%         for offset and scaling
for i = 1:N

    % --- 1) subject-specific template (TDC: LOO) ---
    if isTDC(i)
        template_i = (tdcSum - INT(i,:)) / (tdcN - 1);
    else
        template_i = template_full;
    end

    % --- 2) subject-specific ordering ---
    [templateSorted, sortIdx] = sort(template_i(:), 'ascend');   % [360 x 1]
    y = INT(i, sortIdx)';                                        % [360 x 1]

    good = isfinite(y) & isfinite(templateSorted);
    if nnz(good) < 50
        continue;
    end

    % --- 3) regression: y = alpha + beta*template + resid ---
    X = [ones(P,1), templateSorted];
    b = X(good,:) \ y(good);

    alpha_i = b(1);
    beta_i  = b(2);

    yhat  = X*b;
    resid = y - yhat;

    metrics.alpha(i) = alpha_i;
    metrics.beta(i)  = beta_i;
    metrics.rms(i)  = sqrt(mean(resid(good).^2, 'omitnan'));
end
%% Principal component analysis of AASP subscales
% PCA is performed after z-score normalization.
% Principal component scores are used in subsequent analyses.
aasp_label = {'Low_Registration','Sensation_Seeking',...
    'Sensory_Sensitivity','Sensation_Avoiding'};
usePCA = true;
if usePCA
    
    X = AASP;   % N x 4
    
    ok = all(isfinite(X),2);
    Xok = X(ok,:);
    
    Xz = zscore(Xok);
    
    % PCA
    [coeff, score, ~, ~, explained] = pca(Xz);
    
    disp('Explained variance (%)');
    disp(explained');
    
    loadingTbl = array2table(coeff, 'VariableNames', compose('PC%d',1:4), ...
                                   'RowNames', aasp_label);
    disp('Loadings (coeff):');
    disp(loadingTbl);
    AASP = score;
else
    aaspNames = aasp_label;
end
%% Correlation analysis
% Spearman correlations between deviation metrics
% and AASP scores.
corrRes = table();

for m = 1:numel(metricNames)
    mn = metricNames{m};
    x = metrics.(mn);

    for a = 1:4
        y = AASP(:,a);

        ok = isfinite(x) & isfinite(y);
        if nnz(ok) < 20
            r = NaN; p = NaN;
        else
            [r,p] = corr(x(ok), y(ok), 'Type','Spearman');
        end

        corrRes = [corrRes; table(string(mn), string(aaspNames{a}), r, p, nnz(ok), ...
            'VariableNames', {'Metric','AASP','r','p','N'})]; %#ok<AGROW>
    end
end

%% Partial correlation analysis
% Controls for covariates specified in 'covar'.
if ~isempty(covar)
    pcorrRes = table();
    for m = 1:numel(metricNames)
        mn = metricNames{m};
        x = metrics.(mn);

        for a = 1:4
            y = AASP(:,a);

            Z = covar;
            ok = isfinite(x) & isfinite(y) & all(isfinite(Z),2);
            if nnz(ok) < (20 + size(Z,2))
                r = NaN; p = NaN;
            else
                % Pearson partial correlation
                % Apply tiedrank() beforehand if a rank-based partial correlation is desired.
                [r,p] = partialcorr(x(ok), y(ok), Z(ok,:));
            end

            pcorrRes = [pcorrRes; table(string(mn), string(aaspNames{a}), r, p, nnz(ok), ...
                'VariableNames', {'Metric','AASP','partial_r','p','N'})]; %#ok<AGROW>
        end
    end

    disp('--- Partial correlation results (controlling covariates) ---');
    disp(pcorrRes);
end

disp('--- Correlation results ---');
disp(corrRes);

metricTable = table((1:N)', isTDC(:), 'VariableNames', {'subjID','isTDC'});
for m = 1:numel(metricNames)
    metricTable.(metricNames{m}) = metrics.(metricNames{m});
end
for a = 1:4
    metricTable.(aaspNames{a}) = AASP(:,a);
end
if ~isempty(covar)
    for k = 1:size(covar,2)
        metricTable.(covarNames{k}) = covar(:,k);
    end
end
%% Covariate residualization
% Removes linear effects of covariates before
% computing Spearman correlations.
% Z = covar;   
% 
% metric_names = fieldnames(metrics);
% nMetric = numel(metric_names);
% nPC = size(AASP,2);
% nRow = nMetric * nPC;
% 
% % type setting
% Result = table( ...
%     strings(nRow,1), ...   % Metric
%     zeros(nRow,1), ...     % PC
%     nan(nRow,1), ...       % rho
%     nan(nRow,1), ...       % p
%     zeros(nRow,1), ...     % N
%     'VariableNames', {'Metric','PC','rho','p','N'});
% row = 1;
% for m = 1:nMetric
%     Y = metrics.(metric_names{m});
% 
%     % remove covariance
%     Yr = residualize(Y, Z);
% 
%     for k = 1:nPC
%         Xk = AASP(:,k);
%         Xkr = residualize(Xk, Z);
% 
%         v = isfinite(Xkr) & isfinite(Yr);
% 
%         if sum(v) > 10
%             [rho,p] = corr(Xkr(v), Yr(v), 'Type','Spearman');
%         else
%             rho = NaN; p = NaN;
%         end
% 
%         Result.Metric{row,1} = metric_names{m};
%         Result.PC(row,1)     = k;
%         Result.rho(row,1)    = rho;
%         Result.p(row,1)      = p;
%         Result.N(row,1)      = sum(v);
% 
%         figure, scatter(Xkr, Yr)
%         row = row + 1;
%     end
% end
% 
% disp(Result);
end
%% ==============================================================
function r = residualize(y, Z)
    % y: [N x 1]
    % Z: [N x K] covariates
    ok = all(isfinite([y Z]),2);
    r = nan(size(y));
    if sum(ok) > size(Z,2) + 2
        b = [ones(sum(ok),1) Z(ok,:)] \ y(ok);
        yhat = [ones(sum(ok),1) Z(ok,:)] * b;
        r(ok) = y(ok) - yhat;
    end
end
