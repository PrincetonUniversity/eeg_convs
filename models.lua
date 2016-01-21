require 'nn'

local M = {}

--expect egInputBatch to have dimensions = [examples, time, features]
M.createFullyConnectedNetwork = function(egInputBatch, numHiddenUnits, 
		numHiddenLayers, numOutputClasses)

	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
	local numTotalFeatures = numTimePoints * numInputUnits
	assert(egInputBatch and numHiddenUnits and numHiddenLayers and numOutputClasses)
	assert(numHiddenLayers >= 0)

	local model = nn.Sequential()
	model:add(nn.View(-1):setNumInputDims(2)) --flatten

	if numHiddenLayers > 1 then
		local lastLayer = numTotalFeatures
		for hiddenLayerIdx = 1, numHiddenLayers do
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


--expect egInputBatch to have dimensions = [examples, time, features]
M.createMaxTempConvClassificationNetwork = function(egInputBatch, numHiddenUnits, 
		numHiddenLayers, numOutputClasses)

	local numTimePoints = egInputBatch:size(2)
	local numInputUnits = egInputBatch:size(3)
	assert(egInputBatch and numHiddenUnits and numHiddenLayers and numOutputClasses)
	assert(numHiddenLayers >= 1)
	assert(numHiddenUnits >= numOutputClasses, 
		'Not advisable to have fewer than numOutputClasses hidden units.')
	print('numOutputClasses' .. numOutputClasses)
	local model = nn.Sequential()
	tempConv = nn.TemporalConvolution(numInputUnits, numHiddenUnits, 1, 1)

	model:add(tempConv)
	model:add(nn.ReLU())
	maxPool = nn.TemporalMaxPooling(numTimePoints, 1)
	model:add(maxPool)

	-- flattens from batch x 1 x numHiddens --> batch numHiddens
	model:add(nn.View(-1):setNumInputDims(2)) 

	--we only want to ReLU() the output if we have hidden layers, otherwise we 
	--want linear output (aka what we already get from the conv output) that will 
	--eventually get sent to a criterion which takes the log soft max using linear 
	--output 
	--TODO: Might want to reconsider this behavior, why not have 
	--conv --> pool --> ReLU --> sigmoid?

	if numHiddenLayers > 1 then
		for hiddenLayerIdx = 1, numHiddenLayers-1 do
			model:add(nn.Linear(numHiddenUnits,numHiddenUnits))
			model:add(nn.ReLU())
		end
	end

	model:add(nn.Linear(numHiddenUnits,numOutputClasses))

	--finally logsoftmax gives us 1 numOutputClasses-way classifier
	model:add(nn.LogSoftMax())

	--local criterion = nn.CrossEntropyCriterion()
	local criterion = nn.ClassNLLCriterion()

	return model, criterion
end


--expect egInputBatch to have dimensions = [examples, time, features]
M.createNoMaxTempConvClassificationNetwork = function(...)
  local args, egInputBatch, numHiddenUnits, numPostConvHiddenLayers, 
      numOutputClasses = dok.unpack(
      {...},
      'createNoMaxTempConvClassificationNetwork',
      'Make a convolution ',
      {arg='egInputBatch',type='Tensor',help='', req=true},
      {arg='numHiddenUnits',type='number',help='num filters in conv and how ' .. 
        ' many hidden units in subsequent hidden layers', req=true},
      {arg='numPostConvHiddenLayers',type='number',help='number of hidden ' .. 
        'layers excluding the output filters we get when we do our conv', 
        req=true},
      {arg='numOutputClasses',type='number',help='', req=false, default=nil}
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

	local model = nn.Sequential()
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


return M
