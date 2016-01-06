local M = {}

M.validationLoss = function(fullState)
	if not fullState.validSetLoss then
		fullState:add('validSetLoss', torch.FloatTensor(args.maxTrainingIterations):fill(-1.0), true)
	end
	local modelOut, targets
	modelOut = fullState.network:forward(fullState.data:getValidData())
	targets = fullState.data:getValidTargets()
	fullState.validSetLoss[fullState.trainingIteration] = fullState.criterion:forward(modelOut, targets)
end


M.testClassAcc = function(fullState, num_classes)
	--unlike validSetLoss, we only do this once at the end
	if not fullState.testAvgClassAcc then
		fullState:add('testAvgClassAcc', torch.FloatTensor{-1.0}, true)
	end
	local confMatrix = optim.ConfusionMatrix(num_classes)

	local modelOut, targets
	modelOut = fullState.network:forward(fullState.data:getTestData())
	targets = fullState.data:getTestTargets()
	confMatrix:zero()
	confMatrix:batchAdd(modelOut,targets)
	confMatrix:updateValids()
	fullState.testAvgClassAcc[1] = confMatrix.totalValid
end



M.testLoss = function(fullState)
	--unlike validSetLoss, we only do this once at the end
	if not fullState.testSetLoss then
		fullState:add('testSetLoss', torch.FloatTensor{-1.0}, true)
	end

	local modelOut, targets
	modelOut = fullState.network:forward(fullState.data:getTestData())
	targets = fullState.data:getTestTargets()

	fullState.testSetLoss[1] = fullState.criterion:forward(modelOut, targets)
end

M.saveForSNRSweep = function (fullState)
	local matio = require 'matio'
	-- we want to get
	local output = {}
	output.trainLoss = fullState.trainSetLoss
	output.testLoss = fullState.testSetLoss
	output.trainClassAcc = fullState.trainAvgClassAcc
	output.testClassAcc = fullState.testAvgClassAcc
	output.args = sleep_eeg.utils.copyArgsRemoveUnsupportedMatioTypes(fullState.args)
	--TODO: see if we can save a table somehow!
	--local shortenedArgs = {}
	--shortenedArgs.sim_data = fullState.args.sim_data
	--output.args = shortenedArgs
	local matFileOut = sleep_eeg.utils.replaceTorchSaveWithMatSave(fullState.args.save_file)
	matio.save(matFileOut, output)
	print('Saved .mat file to: ' .. matFileOut)
	print('____________________________________________________')
	print('Final Train Class Acc: '  .. output.trainClassAcc[-1])
	print('Final Test Class Acc: '  .. output.testClassAcc[-1])
end

M.saveForRNGSweep = function(fullState)
	local matio = require 'matio' 
	local output ={}
	output.trainLoss = fullState.trainSetLoss
	output.validLoss = fullState.validSetLoss
	output.trainClassAcc = fullState.trainAvgClassAcc
	output.validClassAcc = fullState.validAvgClassAcc
	if fullState.trainAvgClassAccSubset then
		output.trainAvgClassAccSubset = fullState.trainAvgClassAccSubset
	end
	if fullState.validAvgClassAccSubset then
		output.validAvgClassAccSubset = fullState.validAvgClassAccSubset
	end
	local matFileOut = sleep_eeg.utils.replaceTorchSaveWithMatSave(fullState.args.save_file)
	matio.save(matFileOut, output)
	print('Saved .mat file to: ' .. matFileOut)
	print('____________________________________________________')
	print('Final Train Class Acc: '  .. output.trainClassAcc[-1])
	print('Final Valid Class Acc: '  .. output.validClassAcc[-1])

end

M.__getConfusionMatrixName = function(trainValidOrTestData)
	assert(trainValidOrTestData and type(trainValidOrTestData) == 'string')
	assert(trainValidOrTestData == 'train' or trainValidOrTestData == 'test' or 
		trainValidOrTestData == 'valid', 'Only valid values are "train", ' ..
		' "valid" or "test"')
	local confMatrixKeyName = trainValidOrTestData .. '_confMatrix'
	return confMatrixKeyName
end

M.subsetConfusionMatrix = function(fullState, ...)
	assert(fullState and torch.type(fullState) == 'sleep_eeg.State',
		'Must pass in an object of type sleep_eeg.State')
	local args, trainValidOrTestData, allClassNames, subsetClassIdx = dok.unpack(
	  {...},
	  'subsetConfusionMatrix',
	  'Makes a hook for a confusion matrix that ignores certain outputs',
	  {arg='trainValidOrTestData', type ='string', help='hook for "train", "valid" or "test" set',req = true},
	  {arg='allClassNames', type ='table', help='table of class names',req = true},
	  {arg='subsetClassIdx', type ='table', help='list-like table of class indexes to keep',req = true}
	)

	--this is for the case where we're training a classifier on multiple classes, but 
	--we just want to consider the accuracy for a subset of those classes
	local confMatrixKeyName = M.__getConfusionMatrixName(trainValidOrTestData) .. '_subset'

	if not fullState[confMatrixKeyName] then
		fullState[confMatrixKeyName] = optim.SubsetConfusionMatrix(allClassNames, subsetClassIdx)
	end
	M.__updateConfusionMatrix(fullState, trainValidOrTestData, confMatrixKeyName, true)

end

M.confusionMatrix = function(fullState, trainValidOrTestData, classNames)
	local optim = require 'optim'
	trainValidOrTestData = trainValidOrTestData or 'train' --valid values = 'train', 'test', 'valid'
	local confMatrixKeyName = M.__getConfusionMatrixName(trainValidOrTestData)

	if not fullState[confMatrixKeyName] then
		if classNames then
			fullState[confMatrixKeyName] = optim.ConfusionMatrix(classNames)
		else
			error('This should never get here - we can fix this later')
			fullState[confMatrixKeyName] = optim.ConfusionMatrix()
		end
	end

	M.__updateConfusionMatrix(fullState, trainValidOrTestData, confMatrixKeyName, false)
end

M.__updateConfusionMatrix = function(fullState, trainValidOrTestData, confMatrixKeyName, isSubset)
	local suffix = ''
	if isSubset then 
		suffix = "Subset"
	end

	--here we actually look into fullState and get it's output, we're breaking generality
	--here just to get this done
	fullState[confMatrixKeyName]:zero()
	local modelOut, targets
	if trainValidOrTestData == 'train' then
		modelOut = fullState.network:forward(fullState.data:getTrainData())
		targets = fullState.data:getTrainTargets()
		local trainAvgClassAccKey = 'trainAvgClassAcc' .. suffix
		if not fullState[trainAvgClassAccKey] then
			fullState:add(trainAvgClassAccKey, torch.FloatTensor(fullState.args.training.maxTrainingIterations):fill(-1.0), true)
		end
	elseif trainValidOrTestData == 'test' then
		modelOut = fullState.network:forward(fullState.data:getTestData())
		targets = fullState.data:getTestTargets()
	elseif trainValidOrTestData == 'valid' then
		modelOut = fullState.network:forward(fullState.data:getValidData())
		targets = fullState.data:getValidTargets()
		local validAvgClassAccKey = 'validAvgClassAcc' .. suffix
		if not fullState[validAvgClassAccKey] then
			fullState:add(validAvgClassAccKey, torch.FloatTensor(fullState.args.training.maxTrainingIterations):fill(-1.0), true)
		end
	else
		error('Invalid value passed as for "trainValidOrTestData"' ..
				'acceptable values are: "train" "test" or "valid"')
	end
	fullState[confMatrixKeyName]:batchAdd(modelOut, targets)

	if trainValidOrTestData == 'valid' then
		fullState[confMatrixKeyName]:updateValids() --update confMatrix

		local validAvgClassAccKey = 'validAvgClassAcc' .. suffix
		fullState[validAvgClassAccKey][fullState.trainingIteration] = fullState[confMatrixKeyName].totalValid

		if fullState.trainingIteration %100 == 0 then
			print('Valid accuracy: ' .. fullState[confMatrixKeyName].totalValid)
		end
	end

	if trainValidOrTestData == 'train' then
		fullState[confMatrixKeyName]:updateValids() --update confMatrix

		local trainAvgClassAccKey = 'trainAvgClassAcc' .. suffix
		fullState[trainAvgClassAccKey][fullState.trainingIteration] = fullState[confMatrixKeyName].totalValid

		if fullState.trainingIteration % 100 == 0 then
			print('Training accuracy: ' .. fullState[trainAvgClassAccKey][fullState.trainingIteration])
		end
	end
end

return M
