function pred = bartPredict(model, x_new)
%BARTPREDICT Posterior predictions at new covariate values.
%
%   pred = bartPredict(model, x_new)
%
% MODEL is the output of WBART, PBART, LBART or MBART (fitted with
% opts.savetrees = true, the default).  Each stored posterior draw of the
% ensemble is evaluated at x_new, so PRED contains the full posterior
% distribution of the fit, not just a point estimate.
%
% Fields of PRED
%   yhat         ndpost-by-nnew draws on the linear/latent scale
%   yhat_mean    posterior mean
%   prob         draws of P(Y=1|x) for pbart/lbart (empty for wbart)
%   prob_mean    posterior mean probability
%   For mbart:   prob is ndpost-by-nnew-by-J, prob_mean is nnew-by-J, and
%                pred_level gives the most probable level per row.
%
% Note that survival predictions come from SURVBART's own time grid: call
% bartPredict(model.fit, [t, x]) with t drawn from model.times, or simply pass
% x_test to SURVBART when fitting.

if strcmp(model.type, 'mbart')
    pred = predictMulti(model, x_new);
    return;
end
if ~isfield(model, 'trees') || isempty(model.trees)
    error('bartPredict:notrees', ...
          'The model has no saved trees; refit with opts.savetrees = true.');
end

Xb = bartBinX(x_new, model.cuts);
nd = numel(model.trees);
nn = size(Xb, 1);
f  = zeros(nd, nn);
for d = 1:nd
    td = model.trees{d};
    fd = zeros(nn, 1);
    for j = 1:numel(td)
        t  = td{j};
        fd = fd + t.mu(bartAssign(t, Xb));
    end
    f(d, :) = fd';
end

pred        = struct();
pred.yhat   = model.offset + f;
pred.yhat_mean = mean(pred.yhat, 1);
switch model.type
    case 'wbart'
        pred.prob = []; pred.prob_mean = [];
    case 'pbart'
        pred.prob      = bartPhi(pred.yhat);
        pred.prob_mean = mean(pred.prob, 1);
    case 'lbart'
        pred.prob      = 1 ./ (1 + exp(-pred.yhat));
        pred.prob_mean = mean(pred.prob, 1);
    otherwise
        error('bartPredict:type', 'Unknown model type "%s".', model.type);
end
end

function pred = predictMulti(model, x_new)
J  = numel(model.levels);
nn = size(x_new, 1);

% Number of draws: taken from the first non-degenerate conditional fit.
nd = 0;
for j = 1:(J - 1)
    fj = model.fits{j};
    if ~(isfield(fj, 'degenerate') && fj.degenerate)
        nd = numel(fj.trees);
        break;
    end
end
if nd == 0
    nd = 1;   % every conditional model was degenerate
end

cp = zeros(nd, nn, J - 1);
for j = 1:(J - 1)
    fj = model.fits{j};
    if isfield(fj, 'degenerate') && fj.degenerate
        cp(:, :, j) = fj.p;
    else
        pj = bartPredict(fj, x_new);
        cp(:, :, j) = pj.prob;
    end
end
prob  = zeros(nd, nn, J);
carry = ones(nd, nn);
for j = 1:(J - 1)
    prob(:, :, j) = cp(:, :, j) .* carry;
    carry         = carry .* (1 - cp(:, :, j));
end
prob(:, :, J) = carry;

pred            = struct();
pred.prob       = prob;
pred.prob_mean  = reshape(mean(prob, 1), [nn, J]);
[~, im]         = max(pred.prob_mean, [], 2);
pred.pred_level = model.levels(im);
pred.yhat       = [];
pred.yhat_mean  = [];
end
