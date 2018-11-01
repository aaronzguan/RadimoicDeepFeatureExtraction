function deepFeatures = extract_deepFeatures(imgs)

    addpath('alexnet');
    net = alexnet;
    net.Layers;
    images = {};
    inputSize = net.Layers(1).InputSize;
    for i = 1:length(imgs)
        image = imresize(imgs{i},inputSize(1:2));
        images{i} = image;
    end

    images = table(images');
    layer = 'fc7';
    % deepFeatures = gather(activations(net,images,layer,'OutputAs','rows','ExecutionEnvironment','gpu'));
    deepFeatures = activations(net,images,layer,'OutputAs','rows');

end


