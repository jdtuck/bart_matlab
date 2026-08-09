classdef bart_model
    %BART_MODEL Bart Model Class

    properties
        model
        samples
    end

    methods
        function obj = bart_model(model)
            obj.model = model;
            obj.samples.sigma = model.sigma;
        end

        function pred = predict(obj,x_new, options)
            arguments
                obj
                x_new
                options.idxSamples = nan;
            end
            idxSamples = options.idxSamples;
            if isnan(idxSamples) 
                idxSamples = 1:length(obj.samples.sigma);
            end

            out = bartPredict(obj.model, x_new);
            pred = out.yhat(idxSamples,:);
        end
    end
end