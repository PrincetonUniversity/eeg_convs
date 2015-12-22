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
	local matFileOut = sleep_eeg.utils.replaceTorchSaveWithMatSave(fullState.args.save_file)
	matio.save(matFileOut, output)
	print('Saved .mat file to: ' .. matFileOut)
	print('____________________________________________________')
	print('Final Train Class Acc: '  .. output.trainClassAcc[-1])
	print('Final Valid Class Acc: '  .. output.validClassAcc[-1])

end

M.confusionMatrix = function(fullState, trainValidOrTestData, classNames)
	require 'optim'
	trainValidOrTestData = trainValidOrTestData or 'train' --valid values = 'train', 'test', 'valid'
	local confMatrixKeyName = trainValidOrTestData .. '_confMatrix'

	if not fullState[confMatrixKeyName] then
		if classNames then
			fullState[confMatrixKeyName] = optim.ConfusionMatrix(classNames)
		else
			error('This should never get here - we can fix this later')
			fullState[confMatrixKeyName] = optim.ConfusionMatrix()
		end
	end
	--here we actually look into fullState and get it's output, we're breaking generality
	--here just to get this done
	fullState[confMatrixKeyName]:zero()
	local modelOut, targets
	if trainValidOrTestData == 'train' then
		modelOut = fullState.network:forward(fullState.data:getTrainData())
		targets = fullState.data:getTrainTargets()
		if not fullState.trainAvgClassAcc then
			fullState:add('trainAvgClassAcc', torch.FloatTensor(fullState.args.training.maxTrainingIterations):fill(-1.0), true)
		end
	elseif trainValidOrTestData == 'test' then
		modelOut = fullState.network:forward(fullState.data:getTestData())
		targets = fullState.data:getTestTargets()
	elseif trainValidOrTestData == 'valid' then
		modelOut = fullState.network:forward(fullState.data:getValidData())
		targets = fullState.data:getValidTargets()
		if not fullState.validAvgClassAcc then
			fullState:add('validAvgClassAcc', torch.FloatTensor(fullState.args.training.maxTrainingIterations):fill(-1.0), true)
		end
	else
		error('Invalid value passed as for "trainValidOrTestData"' ..
				'acceptable values are: "train" "test" or "valid"')
	end
	fullState[confMatrixKeyName]:batchAdd(modelOut, targets)
	if trainValidOrTestData == 'valid' then
		if fullState.trainingIteration %100 == 0 then
			fullState[confMatrixKeyName]:updateValids() --update confMatrix
			
			print('Valid accuracy: ' .. fullState[confMatrixKeyName].totalValid)
		end
		fullState.validAvgClassAcc[fullState.trainingIteration] = fullState[confMatrixKeyName].totalValid
	end

	if trainValidOrTestData == 'train' then
		fullState[confMatrixKeyName]:updateValids() --update confMatrix
		fullState.trainAvgClassAcc[fullState.trainingIteration] = fullState[confMatrixKeyName].totalValid
		if fullState.trainingIteration % 100 == 0 then
			print('Training accuracy: ' .. fullState.trainAvgClassAcc[fullState.trainingIteration])
		end
	end
end

return M
