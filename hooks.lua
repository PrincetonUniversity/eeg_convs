local M = {}

M.OUTPUT_EVERY_X_ITERATIONS = 100

M.validLoss = function(fullState)
	if not fullState.validSetLoss then
		fullState:add('validSetLoss', torch.FloatTensor(fullState.args.training.maxTrainingIterations):fill(-1.0), true)
	end
	local modelOut, targets
	modelOut = fullState.network:forward(fullState.data:getValidData())
	targets = fullState.data:getValidTargets()
	fullState.validSetLoss[fullState.trainingIteration] = fullState.criterion:forward(modelOut, targets)

	if fullState.trainingIteration % M.OUTPUT_EVERY_X_ITERATIONS == 0 then
		print('Validation Loss: ' .. fullState.validSetLoss[fullState.trainingIteration])
	end
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

--training iteration hook
M.logWeightToUpdateNormRatio = function(fullState)
	--only works with optim.adam_log or optim.sgd_log
	if not fullState.weightToUpdateNormRatio then
		fullState:add('weightToUpdateNormRatio', torch.DoubleTensor(fullState.args.training.maxTrainingIterations):fill(-1.0), true)
	end
	fullState.weightToUpdateNormRatio[fullState.trainingIteration] = fullState.params:norm()/fullState.optimSettings.update_norm
end

M.randomClassAcc = function(fullState, num_classes)
	local randomData = fullState.data:getTestData():clone():normal(0,3)
	local randomTargets = fullState.data:getTestTargets():clone():random(1,num_classes)

	local confMatrix = optim.ConfusionMatrix(num_classes)
	confMatrix:zero()

	for i = 1,3 do
		randomData:normal(0,3)
		randomTargets:random(1,num_classes)
		local modelOut = fullState.network:forward(randomData)
		confMatrix:batchAdd(modelOut,randomTargets)
	end

	confMatrix:updateValids()
	if not fullState.randomClassAcc then
		fullState:add('randomClassAcc', confMatrix.totalValid, true)
	else
		fullState.randomClassAcc = confMatrix.totalValid
	end
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

M.lookAtDistributionOfMaxValues = function()
	error('not yet implemented')
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
	print('Final Train Class Acc: '  .. output.trainClassAcc[fullState.trainingIteration])
	print('Final Test Class Acc: '  .. output.testClassAcc[fullState.trainingIteration])
end

M.saveForRNGSweep = function(fullState)
	local matio = require 'matio' 
	local output ={}
	output.trainLoss = fullState.trainSetLoss
	output.validLoss = fullState.validSetLoss
	output.trainClassAcc = fullState.trainAvgClassAcc
	output.validClassAcc = fullState.validAvgClassAcc
	--save actual confusion matrix
	output.trainConfMatrix = fullState[M.__getConfusionMatrixName('train')].mat
	output.validConfMatrix = fullState[M.__getConfusionMatrixName('valid')].mat
	if fullState.args.subj_data.run_single_subj then
		output.subj_id = fullState.data:getSubjID()
	end

	if fullState.trainAvgClassAccSubset then
		output.trainAvgClassAccSubset = fullState.trainAvgClassAccSubset
		output.trainConfMatrixSubset = fullState[M.__getConfusionMatrixName('train') .. '_subset'].mat
	end

	if fullState.validAvgClassAccSubset then
		output.validAvgClassAccSubset = fullState.validAvgClassAccSubset
		output.validConfMatrixSubset = fullState[M.__getConfusionMatrixName('valid') .. '_subset'].mat
	end

	if fullState.randomClassAcc then
		output.randomClassAcc = torch.FloatTensor{fullState.randomClassAcc}
	end

	--save weight to update ratio
	output.weightToUpdateNormRatio = fullState.weightToUpdateNormRatio

	local newSaveFile
	if fullState.args.subj_data and fullState.args.subj_data.run_single_subj then
		newSaveFile = sleep_eeg.utils.insertDirToSaveFile(fullState.args.save_file, fullState.data:getSubjID())
	else
		newSaveFile = fullState.args.save_file
	end

	local matFileOut = sleep_eeg.utils.replaceTorchSaveWithMatSave(newSaveFile)
	matio.save(matFileOut, output)
	print('Saved .mat file to: ' .. matFileOut)
	print('____________________________________________________')
	print('Final Train Class Acc: '  .. output.trainClassAcc[fullState.trainingIteration])
	print('Final Valid Class Acc: '  .. output.validClassAcc[fullState.trainingIteration])

end

M.plotForRNGSweep = function(fullState)
	require 'gnuplot'
	
	--helper fns
	local plotSymbol = function(plotTable, name, values) 
		table.insert(plotTable, {name, values, '-'})
	end

	local makeAndSavePlot = function(saveFile, title, plots)
		local pngfig = gnuplot.pngfigure(saveFile)
		gnuplot.plot(plots)
		gnuplot.grid('on')
		gnuplot.title(title)
		gnuplot.plotflush()
		gnuplot.close(pngfig)
	end

	--make two plots: one for losses, one for classification accuracy
	--loss plots
	local lossPlots = {}
	plotSymbol(lossPlots, 'Train Loss', fullState.trainSetLoss)
	plotSymbol(lossPlots, 'Valid Loss', fullState.validSetLoss)
	local newSaveFile = sleep_eeg.utils.insertDirToSaveFile(fullState.args.save_file, fullState.data:getSubjID())
	local saveFile = sleep_eeg.utils.replaceTorchSaveWithPngSave(newSaveFile, 'Losses')
	print('Saving plot to: ' .. saveFile)
	makeAndSavePlot(saveFile, 'Losses', lossPlots)
	
	--class acc plots
	local classAccPlots = {}
	plotSymbol(classAccPlots, 'Train Acc', fullState.trainAvgClassAcc)
	plotSymbol(classAccPlots, 'Valid Acc', fullState.validAvgClassAcc)
	
	if fullState.trainAvgClassAccSubset then
		plotSymbol(classAccPlots, 'Train Subset', fullState.trainAvgClassAccSubset)
	end

	if fullState.validAvgClassAccSubset then
		plotSymbol(classAccPlots, 'Valid Subset', fullState.validAvgClassAccSubset)
	end
	saveFile = sleep_eeg.utils.replaceTorchSaveWithPngSave(newSaveFile, 'ClassAcc')
	print('Saving plot to: ' .. saveFile)
	makeAndSavePlot(saveFile, 'Class Acc', classAccPlots)

	--if fullState.randomClassAcc then
		--output.randomClassAcc = torch.FloatTensor{fullState.randomClassAcc}
	--end

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

		if fullState.trainingIteration % M.OUTPUT_EVERY_X_ITERATIONS == 0 then
			print('Valid accuracy: ' .. fullState[confMatrixKeyName].totalValid)
		end
	end

	if trainValidOrTestData == 'train' then
		fullState[confMatrixKeyName]:updateValids() --update confMatrix

		local trainAvgClassAccKey = 'trainAvgClassAcc' .. suffix
		fullState[trainAvgClassAccKey][fullState.trainingIteration] = fullState[confMatrixKeyName].totalValid

		if fullState.trainingIteration % M.OUTPUT_EVERY_X_ITERATIONS == 0 then
			print('Training accuracy: ' .. fullState[trainAvgClassAccKey][fullState.trainingIteration])
		end
	end
end

return M
