local M = {}

M.SGD = function(learningRate)
	require 'optim'
	if learningRate == nil then
		learningRate = 0.001
	end
	optimSettings = {learningRate = learningRate, momentum = 0.9, decay = 0}
	return optim.sgd, optimSettings
end

M.ADAM = function(learningRate)
	require 'optim'
	if learningRate == nil then
		learningRate = 0.001
	end
	optimSettings = {learningRate = learningRate}
	return optim.adam, optimSettings
end

M.getOptimizer = function(name, learningRate)
	if name == 'sgd' then
		return M.SGD(learningRate)
	elseif name == 'adam' then
		return M.ADAM(learningRate)
	else
		error('Currently only supports name = "sgd" or "adam"')
	end
end

M.performTrainIteration = function(fullState)
	local args = fullState.args.training
	local trainInputs = fullState.data:getTrainData()
	local trainTargets = fullState.data:getTrainTargets()

	if not fullState.trainSetLoss then
		fullState:add('trainSetLoss', torch.FloatTensor(args.maxTrainingIterations):fill(-1.0), true)
	end

	--set gradient to zero
	fullState.gradParams:zero()
   	fullState.trainModelOut = fullState.network:forward(trainInputs)
	fullState.trainSetLoss[fullState.trainingIteration] = fullState.criterion:forward(fullState.trainModelOut, trainTargets)
    fullState.trainSetClassAcc = 1 --evaluation.classification(trainModelOut, trainTargets)

    --actually update our network
    fullState.network:backward(trainInputs, fullState.criterion:backward(fullState.trainModelOut, trainTargets))

    fullState.optimizer(function() return fullState.trainSetLoss, fullState.gradParams end,
        fullState.params, fullState.optimSettings)

	if fullState.trainingIteration % 100 == 0 then
		print('_________________________________')
		print('----ITERATION: ' .. fullState.trainingIteration .. '-------')
		print('Training Loss: ')
		print(fullState.trainSetLoss[fullState.trainingIteration])
		print('')
	end
end

return M
