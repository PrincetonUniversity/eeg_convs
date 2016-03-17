require 'nn'

local M = {}

--expect egInputBatch to have dimensions = [examples, time, features]
M.createFullyConnectedNetwork = function(egInputBatch, numHiddenUnits, 
		numHiddenLayers, numOutputClasses, dropout_prob, predict_subj, 
		numSubjects, net_args)

	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
	local numTotalFeatures = numTimePoints * numInputUnits
	assert(egInputBatch and numHiddenUnits and numHiddenLayers and numOutputClasses)
	assert(numHiddenLayers >= 0)
	dropout_prob = dropout_prob or -1

    if predict_subj then
		require 'nngraph'

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
		prev = nn.View(-1):setNumInputDims(2)(prev)
		local toClasses = {}
		local toSubjects = {}
		if numHiddenLayers > 1 then
			local lastLayer = numTotalFeatures
			for hiddenLayerIdx = 1, numHiddenLayers-1 do
				prev = nn.Linear(lastLayer,numHiddenUnits)(prev)
				prev = nn.ReLU()(prev)
				lastLayer = numHiddenUnits
			end

			--now we split
			toClasses = nn.Linear(numHiddenUnits,numOutputClasses)(prev)
			toSubjects = nn.Linear(numHiddenUnits, numSubjects)(prev)
		else

			toClasses = nn.Linear(numTotalFeatures,numOutputClasses)(prev)
			toSubjects = nn.Linear(numTotalFeatures, numSubjects)(prev)
		end
		toClasses = nn.LogSoftMax()(toClasses)
		toSubjects = nn.LogSoftMax()(toSubjects)
		model = nn.gModule({input},{toClasses, toSubjects})

		criterion = nn.ParallelCriterion()
		--weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		--model:forward(egInputBatch)
		--graph.dot(model.fg, 'mlp','test2')

		return model, criterion
	else
		local model = nn.Sequential()
		if dropout_prob > 0 then
			model:add(nn.Dropout(dropout_prob))
		end
		model:add(nn.View(-1):setNumInputDims(2)) --flatten

		if numHiddenLayers > 1 then
			local lastLayer = numTotalFeatures
			for hiddenLayerIdx = 1, numHiddenLayers-1 do
				model:add(nn.Linear(lastLayer,numHiddenUnits))
				model:add(nn.ReLU())
				lastLayer = numHiddenUnits
			end

			model:add(nn.Linear(numHiddenUnits,numOutputClasses))
		else
			model:add(nn.Linear(numTotalFeatures,numOutputClasses))
		end

		--finally logsoftmax gives us 1 numOutputClasses-way classifier
		model:add(nn.LogSoftMax())

		--local criterion = nn.CrossEntropyCriterion()
		local criterion = nn.ClassNLLCriterion()

		return model, criterion
	end

end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createSumTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createSumTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  if predict_subj then
		require 'nngraph'

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
    prev = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)(prev)
    prev = nn.ReLU()(prev)
    --sum, instead of max, across temporal dimension
    prev = nn.Sum(2)(prev) --only works if we have batch data!!!!
    --usually we need a nn:View(-1) to collapse the singleton temporal dimension, but sum gets rid of that

    local prevLayerOutputs =  numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = nn.ReLU()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    model = nn.gModule({input},{toClasses, toSubjects})

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		model:forward(egInputBatch)
		graph.dot(model.fg, 'mlp','test_max_temp_conv')

    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end
    tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(nn.ReLU())
    out = model:forward(egInputBatch)
    --sum, instead of max, across temporal dimension
    model:add(nn.Sum(2)) --only works if we have batch data!!!
    --usually we need a nn:View(-1) to collapse the singleton temporal dimension, but sum gets rid of that

    --we only want to ReLU() the output if we have hidden layers, otherwise we 
    --want linear output (aka what we already get from the conv output) that will 
    --eventually get sent to a criterion which takes the log soft max using linear 
    --output 
    --TODO: Might want to reconsider this behavior, why not have 
    --conv --> pool --> ReLU --> sigmoid?
    local prevLayerOutputs = numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(nn.ReLU())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes
    model:add(nn.Linear(prevLayerOutputs,numOutputClasses))

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    model:add(nn.LogSoftMax())

    --local criterion = nn.CrossEntropyCriterion()
    local criterion = nn.ClassNLLCriterion()

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createShallowMaxTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createShallowMaxTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2


  if predict_subj then
    error("-predict_subj not supported for shallow_temp_conv");
  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end

    tempConv = nn.TemporalConvolution(numInputUnits, numOutputClasses, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(nn.TemporalMaxPooling(numTimePoints,1))
    model:add(nn.View(-1):setNumInputDims(2)) 

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    model:add(nn.LogSoftMax())

    --local criterion = nn.CrossEntropyCriterion()
    local criterion = nn.ClassNLLCriterion()

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createMaxChannelConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createMaxChannelConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
	local numTimePoints = egInputBatch:size(3)
	local numInputUnits = egInputBatch:size(2)
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  if predict_subj then
		require 'nngraph'

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
    prev = nn.Transpose({2,3})
    prev = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)(prev)
    prev = nn.ReLU()(prev)
    prev = nn.TemporalMaxPooling(numTimePoints, 1)(prev)
    prev = nn.View(-1):setNumInputDims(2)(prev)

    local prevLayerOutputs =  numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = nn.ReLU()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    model = nn.gModule({input},{toClasses, toSubjects})

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		model:forward(egInputBatch)
		graph.dot(model.fg, 'mlp','test_max_temp_conv')

    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end
    model:add(nn.Transpose({2,3}))
    tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(nn.ReLU())
    model:add(nn.TemporalMaxPooling(numTimePoints,1))
    model:add(nn.View(-1):setNumInputDims(2))

    --we only want to ReLU() the output if we have hidden layers, otherwise we
    --want linear output (aka what we already get from the conv output) that will 
    --eventually get sent to a criterion which takes the log soft max using linear 
    --output 
    --TODO: Might want to reconsider this behavior, why not have 
    --conv --> pool --> ReLU --> sigmoid?
    local prevLayerOutputs = numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(nn.ReLU())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes
    model:add(nn.Linear(prevLayerOutputs,numOutputClasses))

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    model:add(nn.LogSoftMax())

    --local criterion = nn.CrossEntropyCriterion()
    local criterion = nn.ClassNLLCriterion()

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createMaxTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createMaxTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
  local smooth_std = net_args.smooth_std or -1
  local smooth_width = net_args.smooth_width or 5

	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  local smoothModule = {}
  local shouldSmooth = false
  if smooth_std > 0 then
    local filter = nn.TemporalSmoothing.filters.makeGaussian(smooth_width, smooth_std, true)
    smoothModule = nn.TemporalSmoothing(filter, net_args.smooth_step, false, 2)
    shouldSmooth = true
  end

  if predict_subj then
		require 'nngraph'

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
    prev = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)(prev)
    prev = nn.ReLU()(prev)
    local prevOutputWidth = numTimePoints
    if shouldSmooth then
      prev = smoothModule(prev)
      prevOutputWidth = smoothModule:getTemporalOutputSize(numTimePoints)
    end
    prev = nn.TemporalMaxPooling(prevOutputWidth, 1)(prev)
    prev = nn.View(-1):setNumInputDims(2)(prev)

    local prevLayerOutputs =  numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = nn.ReLU()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    model = nn.gModule({input},{toClasses, toSubjects})

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		--model:forward(egInputBatch)
		--graph.dot(model.fg, 'mlp','test_max_temp_conv')

    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end
    tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(nn.ReLU())

    local prevOutputWidth = numTimePoints
    if shouldSmooth then
      model:add(smoothModule)
      prevOutputWidth = smoothModule:getTemporalOutputSize(numTimePoints)
    end

    model:add(nn.TemporalMaxPooling(prevOutputWidth,1))
    model:add(nn.View(-1):setNumInputDims(2)) 

    --we only want to ReLU() the output if we have hidden layers, otherwise we 
    --want linear output (aka what we already get from the conv output) that will 
    --eventually get sent to a criterion which takes the log soft max using linear 
    --output 
    --TODO: Might want to reconsider this behavior, why not have 
    --conv --> pool --> ReLU --> sigmoid?
    local prevLayerOutputs = numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(nn.ReLU())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes
    model:add(nn.Linear(prevLayerOutputs,numOutputClasses))

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    model:add(nn.LogSoftMax())

    --local criterion = nn.CrossEntropyCriterion()
    local criterion = nn.ClassNLLCriterion()

    return model, criterion
  end
end

--expect egInputBatch to have dimensions = [examples, time, features]
M.createNoMaxTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses, dropout_prob, predict_subj, numSubjects, net_args = dok.unpack(
      {...},
      'createNoMaxTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil},
      {arg='dropout_prob',type='number',help='', req=false, default=-1},
      {arg='predict_subj',type='number',
        help='whether or not to predict subjects as well as classes', req=false, default=false},
      {arg='numSubjects',type='number',help='only applies if predict_subj is true', 
        req=false, default=-1},
      {arg='net_args',type='table',help='', req=true}
  )
	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
  print(numTimePoints, numInputUnits)
	assert(egInputBatch and numHiddenUnits and numPostConvHiddenLayers)
	--if we're not going to take the max after our convolution, which collapses
	--the number of output features, then we have to have at least one hidden
	--layer
	assert(numPostConvHiddenLayers > 0)
	numOutputClasses = numOutputClasses or 2

  if predict_subj then
		require 'nngraph'

		local input = nn.Identity()()
		local prev = {}
		if dropout_prob > 0 then
			prev = nn.Dropout(dropout_prob)(input)
		else
			prev = input
		end
    prev = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)(prev)
    prev = nn.View(-1):setNumInputDims(2)(prev)
    prev = nn.ReLU()(prev)

    local prevLayerOutputs =  numTimePoints * numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      prev = nn.Linear(prevLayerOutputs,numHiddenUnits)(prev)
      prev = nn.ReLU()(prev)
      prevLayerOutputs = numHiddenUnits
    end

    --finally go from shared hidden to individual output paths
    local toClasses = nn.Linear(prevLayerOutputs,numOutputClasses)(prev)
    local toSubjects = nn.Linear(prevLayerOutputs, numSubjects)(prev)
    toClasses = nn.LogSoftMax()(toClasses)
    toSubjects = nn.LogSoftMax()(toSubjects)

    model = nn.gModule({input},{toClasses, toSubjects})

    criterion = nn.ParallelCriterion()
    --weight subjects two times as important
		criterion:add(nn.ClassNLLCriterion(),net_args.class_to_subj_loss_ratio)
		criterion:add(nn.ClassNLLCriterion(),1)

		--model:forward(egInputBatch)
		--graph.dot(model.fg, 'mlp','test_no_max_temp_conv')

    return model, criterion

  else

    local model = nn.Sequential()
    if dropout_prob > 0 then
      model:add(nn.Dropout(dropout_prob))
    end
    tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)
    model:add(tempConv)

    -- flattens from batch x 1 x numHiddens --> batch numHiddens
    -- now we have batch x numTimePoints x numHiddens --> batch x numTimePoints * numHiddens
    model:add(nn.View(-1):setNumInputDims(2)) 
    model:add(nn.ReLU())

    --we only want to ReLU() the output if we have hidden layers, otherwise we 
    --want linear output (aka what we already get from the conv output) that will 
    --eventually get sent to a criterion which takes the log soft max using linear 
    --output 
    --TODO: Might want to reconsider this behavior, why not have 
    --conv --> pool --> ReLU --> sigmoid?
    local prevLayerOutputs = numTimePoints * numHiddenUnits --from the convNet

    for hiddenLayerIdx = 2, numPostConvHiddenLayers do
      model:add(nn.Linear(prevLayerOutputs,numHiddenUnits))
      model:add(nn.ReLU())
      prevLayerOutputs = numHiddenUnits
    end

    --go from last hidden layer to number of classes
    model:add(nn.Linear(prevLayerOutputs,numOutputClasses))

    --finally logsoftmax gives us 1 numOutputClasses-way classifier
    model:add(nn.LogSoftMax())

    --local criterion = nn.CrossEntropyCriterion()
    local criterion = nn.ClassNLLCriterion()

    return model, criterion
  end
end

return M
