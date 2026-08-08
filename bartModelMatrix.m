function [X, info] = bartModelMatrix(data, info)
%BARTMODELMATRIX Numeric design matrix for BART, with dummy-coded factors.
%
%   [X, info] = bartModelMatrix(data)
%   X         = bartModelMatrix(newdata, info)
%
% BART needs a numeric matrix of covariates.  Categorical covariates should be
% expanded into one dummy per level (BART does not need a reference level to be
% dropped, because splits are on individual dummies).  This helper does that
% for a table or a cell array of columns, and returns INFO so that new data can
% be encoded exactly the same way.
%
% Accepted inputs
%   numeric matrix            passed through unchanged
%   table                     numeric columns kept; categorical/char/cellstr
%                             and logical columns expanded to dummies
%   1-by-p cell array         each cell a numeric vector or a cellstr/char
%                             column of labels
%
% INFO fields
%   names   1-by-size(X,2) cellstr of column names
%   spec    per input column: type ('num' or 'fac') and level list

if nargin < 2, info = []; end
mkinfo = isempty(info);

% ---- normalize the input to a cell array of columns ----------------------
if isnumeric(data) || islogical(data)
    cols  = num2cell(double(data), 1);
    inames = arrayfun(@(j) sprintf('x%d', j), 1:size(data, 2), ...
                      'UniformOutput', false);
elseif isa(data, 'table')
    vn     = data.Properties.VariableNames;
    cols   = cell(1, numel(vn));
    inames = vn;
    for j = 1:numel(vn)
        cols{j} = data.(vn{j});
    end
elseif iscell(data)
    cols   = data(:)';
    inames = arrayfun(@(j) sprintf('x%d', j), 1:numel(cols), ...
                      'UniformOutput', false);
else
    error('bartModelMatrix:input', ...
          'DATA must be a numeric matrix, a table, or a cell array of columns.');
end

p = numel(cols);
if mkinfo
    info = struct('names', {{}}, 'spec', {cell(1, p)});
end

X     = [];
names = {};
for j = 1:p
    cj = cols{j};
    if isnumeric(cj) || islogical(cj)
        if islogical(cj)
            cj = double(cj);
        end
        if size(cj, 2) > 1
            % already a numeric block
            X = [X, double(cj)];   %#ok<AGROW>
            for c = 1:size(cj, 2)
                names{end+1} = sprintf('%s_%d', inames{j}, c);  %#ok<AGROW>
            end
            if mkinfo, info.spec{j} = struct('type', 'num', 'levels', {{}}, 'width', size(cj,2)); end
            continue;
        end
        X = [X, double(cj(:))];    %#ok<AGROW>
        names{end+1} = inames{j};  %#ok<AGROW>
        if mkinfo, info.spec{j} = struct('type', 'num', 'levels', {{}}, 'width', 1); end
    else
        lab = toLabels(cj);
        if mkinfo
            lev = unique(lab);
            info.spec{j} = struct('type', 'fac', 'levels', {lev}, ...
                                  'width', numel(lev));
        end
        lev = info.spec{j}.levels;
        D   = zeros(numel(lab), numel(lev));
        for l = 1:numel(lev)
            D(:, l) = double(strcmp(lab, lev{l}));
        end
        if any(sum(D, 2) == 0)
            warning('bartModelMatrix:newlevel', ...
                    'Column %s has levels unseen in the training data.', inames{j});
        end
        X = [X, D];   %#ok<AGROW>
        for l = 1:numel(lev)
            names{end+1} = sprintf('%s_%s', inames{j}, lev{l});  %#ok<AGROW>
        end
    end
end
info.names = names;
end

function lab = toLabels(c)
if ischar(c)
    lab = cellstr(c);
elseif iscell(c)
    lab = cellfun(@toChar, c, 'UniformOutput', false);
else
    lab = cellstr(char(c));   % categorical and similar
end
lab = lab(:);
end

function s = toChar(v)
if ischar(v)
    s = v;
else
    s = num2str(v);
end
end
