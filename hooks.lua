local M = {}

M.OUTPUT_EVERY_X_ITERATIONS = 100

M.validLoss = function(fullState)
	if not fullState.validSetLoss then
		fullState:add('validSetLoss', torch.FloatTensor(fullState.args.training.maxTrainingIterations):fill(-1.0), true)
	end


  local data = fullState.data:getValidData()
  local targets = fullState.data:getValidTargets()
  local numExamples = data:size(1)
  local numMiniBatches = sleep_eeg.utils.getNumMiniBatches(numExamples, fullState.args.miniBatchSize)

  fullState.validSetLoss[fullState.trainingIteration] = 0

  for miniBatchIdx = 1, numMiniBatches do

    local miniBatchTrials = sleep_eeg.utils.getMiniBatchTrials(torch.range(1,numExamples):long(), miniBatchIdx, fullState.args.miniBatchSize)
    local miniBatchWeight = miniBatchTrials:numel()/numExamples 

    local modelOut = fullState.network:forward(data:index(1,miniBatchTrials))
    local batch_targets = sleep_eeg.utils.indexIntoTensorOrTableOfTensors(targets,1,miniBatchTrials)

    fullState.validSetLoss[fullState.trainingIteration] = fullState.validSetLoss[fullState.trainingIteration] +  fullState.criterion:forward(modelOut, batch_targets) * miniBatchWeight

  end

  fullState.validSetLoss[fullState.trainingIteration] = fullState.validSetLoss[fullState.trainingIteration]/numMiniBatches

	if fullState.trainingIteration % M.OUTPUT_EVERY_X_ITERATIONS == 0 then
    print('Validation Loss: ' .. fullState.validSetLoss[fullState.trainingIteration])
  end
  if not fullState.plotting then
    fullState.plotting = {}
  end

  if not fullState.plotting.losses then
    fullState.plotting.losses = {}
  end
  if not fullState.plotting.losses.validSetLoss then
    fullState.plotting.losses.validSetLoss = 'validSetLoss'
  end
end

M.decrease_learning_rate = function(fullState)
  if fullState.trainingIteration % fullState.args.training.iterationsDecreaseLR == 0 then
    print(string.format('Decreasing learning rate %0.7f by %0.2f', fullState.optimSettings['learningRate'], fullState.args.training.percentDecreaseLR/100))
    fullState.optimSettings['learningRate'] = fullState.optimSettings['learningRate'] * fullState.args.training.percentDecreaseLR/100
    print(string.format('New learning rate: %0.7f', fullState.optimSettings['learningRate']))
  end
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
  --we don't want to do this
  if fullState.args.subj_data.predict_subj then
    fullState.randomClassAcc = -1
  else
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
end

M.testLoss = function(fullState)
  --unlike validSetLoss, we only do this once at the end
  if not fullState.testSetLoss then
    fullState:add('testSetLoss', torch.FloatTensor( fullState.args.training.maxTrainingIterations ):fill(-1.0), true)
  end

  local modelOut, targets
  modelOut = fullState.network:forward(fullState.data:getTestData())
  targets = fullState.data:getTestTargets()
  fullState.testSetLoss[fullState.trainingIteration] = fullState.criterion:forward(modelOut, targets)
  if fullState.trainingIteration % M.OUTPUT_EVERY_X_ITERATIONS == 0 then
    print('Test Loss: ' .. fullState.testSetLoss[fullState.trainingIteration])
  end
  if not fullState.plotting then
    fullState.plotting = {}
  end

  if not fullState.plotting.losses then
    fullState.plotting.losses = {}
  end

  if not fullState.plotting.testSetLoss then
    fullState.plotting.losses.testSetLoss = 'testSetLoss'
  end
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
	--print('Final Train Class Acc: '  .. output.trainClassAcc[fullState.trainingIteration])
	--print('Final Test Class Acc: '  .. output.testClassAcc[fullState.trainingIteration])
end

M.saveAggregationScript = function(fullState)

  if fullState.aggregationScriptWritten then
    return
  end

  local classAccVarNames = {}
  local confusionMatrices = {}
  local numClassesPerConfMatrix = {}

  --add accuracy and the confusion matrices behind them
  for k,v in pairs(fullState.plotting.accuracy) do
    classAccVarNames[#classAccVarNames+1] = v
    if fullState[k] and fullState[k].mat then --confusion matrix
      confusionMatrices[#confusionMatrices+1] = k
      numClassesPerConfMatrix[#numClassesPerConfMatrix+1] = fullState[k].mat:size(1)
    end
  end

	local newSaveFile
	if fullState.args.subj_data and fullState.args.subj_data.run_single_subj then
		newSaveFile = sleep_eeg.utils.insertDirToSaveFile(fullState.args.save_file, fullState.data:getSubjID())
	else
		newSaveFile = fullState.args.save_file
	end

	local matFileOut = sleep_eeg.utils.replaceTorchSaveWithMatSave(newSaveFile)
 
  --here we write a file that will aggregate any kfold result
  local numFolds = fullState.args.subj_data.num_folds or 12
  if numFolds then
    local aggregationScript = sleep_eeg.utils.removeFoldNumber(sleep_eeg.utils.replaceMinuses(sleep_eeg.utils.replaceTorchSaveWithMSave(newSaveFile)), fullState.args.subj_data.fold_num, fullState.args.subj_data.num_folds)
    local totalTrials = fullState.args.training.maxTrainingIterations
    if not sleep_eeg.utils.doesFileExist(aggregationScript) then
      local outFile = io.open(aggregationScript, 'w')
      outFile:write('\n') --just to get it on disk
      io.close(outFile)

      --here we declare helper functions that return strings of matlab code for loading, 
      --aggregating, averaging, and plotting all the variables we end up saving
      -----------------------------------------------------------------------------------
      --declaring aggregators
      local declare_accuracy_aggregator = function(var_name)
        return string.format('%% dims = # folds, # trials\n%s_agg = zeros(%d,%d);\n\n',var_name, numFolds, totalTrials)
      end

      local declare_conf_matrix_aggregator = function(var_name, numClassesConfMatrix)
        return string.format('%% dims = # folds, # conf matrix classes, # conf matrix classes\n%s_agg = zeros(%d,%d,%d);\n\n',var_name, numFolds, numClassesConfMatrix, numClassesConfMatrix)
      end

      --loading folds results into aggregator
      local aggregate_conf_matrix = function(var_name,fold_idx_var)
        return string.format('%s_agg(%s,:,:) = %s;\n',var_name, fold_idx_var, var_name)
      end

      --this is used to put a variable into the aggregated variable, run inside a for loop over folds
      local aggregate_accuracy = function(var_name, fold_idx_var)
        return string.format('%s_agg(%s,:) = %s;\n',var_name, fold_idx_var, var_name)
      end

      --taking mean of aggregated data across folds
      local collapse_accuracy_and_conf_matrix = function(var_name,fn_name)
        return string.format('%s_%s = squeeze(%s(%s_agg,1));\n',var_name, fn_name, fn_name, var_name)
      end

      local subplot_and_imagesc = function(var_name, subplot_size, subplot_idx)
        return string.format("subplot(%d,%d,%d);\nimagesc_text(%s); title('%s');\n", subplot_size, subplot_size, subplot_idx, var_name, var_name)
      end
      local simple_line_plot = function(var_name)
        return string.format("plot(%s); title('%s');\n", var_name, var_name)
      end

      local save_plot_and_close_figure = function(save_name)
        return string.format("print(gcf,'-dpng','-painters','%s.png');\nclose(gcf);\n\n",save_name)
        --return string.format("export_fig(gcf,'%s.png');\nclose(gcf);\n\n",save_name)
      end

      --reopen it
      outFile = io.open(aggregationScript, 'w')

      --set our aggregate variable declarations
      --comment our auto-generated code
      local codeTemplate = "% let's declare the variable that we'll use to aggregate results over folds\n"

      for idx, var in ipairs(classAccVarNames) do
        codeTemplate = codeTemplate .. declare_accuracy_aggregator(var)
      end

      for idx, var in ipairs(confusionMatrices) do
        codeTemplate = codeTemplate .. declare_conf_matrix_aggregator(var, numClassesPerConfMatrix[idx])
      end

      --write the name of the .mat files for each fold to a cell array so we can iteratively load them
      codeTemplate = codeTemplate .. '\nresult_MAT_files = {'
      local newMatFile = sleep_eeg.utils.replaceTorchSaveWithMatSave(newSaveFile)
      for fold_idx = 1, numFolds do
        codeTemplate = codeTemplate .. "'" .. sleep_eeg.utils.replaceFoldNumber(newMatFile, fullState.args.subj_data.fold_num or 1, fold_idx) .. "' ... \n "
      end
      codeTemplate = codeTemplate .. '};\n'

      codeTemplate = codeTemplate .. string.format('%% now we aggregate over all our folds, loading each saved .mat file in turn\n' ..
        'for fold_idx = 1 : %d;\n\tload(result_MAT_files{fold_idx});\n', numFolds)

      --aggregating code
      for idx, var in ipairs(classAccVarNames) do
        codeTemplate = codeTemplate .. '\t' .. aggregate_accuracy(var, 'fold_idx')
      end

      for idx, var in ipairs(confusionMatrices) do
        codeTemplate = codeTemplate ..'\t' ..  aggregate_conf_matrix(var, 'fold_idx')
      end

      --finish for loop for aggregation
      codeTemplate = codeTemplate .. 'end\n\n'

      --now we take the average and std across the folds
      codeTemplate = codeTemplate .. '% take average and std across folds\n'

      for idx, var in ipairs(classAccVarNames) do
        codeTemplate = codeTemplate  .. collapse_accuracy_and_conf_matrix(var,'mean')
        codeTemplate = codeTemplate  .. collapse_accuracy_and_conf_matrix(var,'std')
      end

      for idx, var in ipairs(confusionMatrices) do
        codeTemplate = codeTemplate  .. collapse_accuracy_and_conf_matrix(var,'mean')
        codeTemplate = codeTemplate  .. collapse_accuracy_and_conf_matrix(var,'std')
      end

      --finally we generate figures:
      --one figure with confusion matrices as imagesc (MEAN)
      --one figure with confusion matrices as imagesc (STD)
      --one figure with the average over all folds
      --one figure with the std over all folds
      --one figure with the per-fold plots for each variable

      codeTemplate = codeTemplate .. string.format("%% add useful toolbox functions;\n addpath(genpath('%s'));\n", sleep_eeg.utils.getMatlabUtilPath())
      codeTemplate = codeTemplate .. "%set plotting defaults\n" .. 
        "set(0,'DefaultFigurePosition',[2100 900 2000 1000],'DefaultLineLineWidth',4,'DefaultAxesFontSize',36,'DefaultTextFontSize',48, 'DefaultFigureVisible', 'off');\n"

      local numConfSubplots = math.ceil(math.sqrt(#confusionMatrices))
      --conf matrix mean
      codeTemplate = codeTemplate .. '\n% plot avg confusion matrices with imagesc\nfigure(1);\n'
      for subplot_idx, var in ipairs(confusionMatrices) do 
        codeTemplate = codeTemplate .. subplot_and_imagesc(var .. '_mean', numConfSubplots, subplot_idx)
      end
      codeTemplate = codeTemplate .. save_plot_and_close_figure('confusionMatrix_means')

      --conf matrix mean
      codeTemplate = codeTemplate .. '\n% plot std confusion matrices with imagesc\nfigure(1);\n'
      for subplot_idx, var in ipairs(confusionMatrices) do 
        codeTemplate = codeTemplate .. subplot_and_imagesc(var .. '_std', numConfSubplots, subplot_idx)
      end
      codeTemplate = codeTemplate .. save_plot_and_close_figure('confusionMatrix_std')

      --plot average across folds
      codeTemplate = codeTemplate .. '\n% plot avg accuracies across folds\nfigure(1); hold all;\n'
      for subplot_idx, var in ipairs(classAccVarNames) do 
        codeTemplate = codeTemplate .. simple_line_plot(var .. '_mean')
      end
      codeTemplate = codeTemplate .. save_plot_and_close_figure('classAcc_mean')

      --plot std across folds
      codeTemplate = codeTemplate .. '\n% plot std accuracies across folds\nfigure(1); hold all;\n'
      for subplot_idx, var in ipairs(classAccVarNames) do 
        codeTemplate = codeTemplate .. simple_line_plot(var .. '_std')
      end
      codeTemplate = codeTemplate .. save_plot_and_close_figure('classAcc_std')

      --one figure with the per-fold plots for each variable
      codeTemplate = codeTemplate .. '\n% plot accuracies for each fold separately\n'
      for subplot_idx, var in ipairs(classAccVarNames) do 
        --create new plot for this variable
        codeTemplate = codeTemplate .. 'figure(1); hold all;\n'
        for fold_idx = 1, numFolds do
          codeTemplate = codeTemplate .. simple_line_plot( string.format("squeeze(%s(%d,:))",var,fold_idx))
        end
        codeTemplate = codeTemplate .. save_plot_and_close_figure(var .. '_all_folds') .. '\n'
      end

      if fullState.args.launch_agg_job then
        --submit job with dependency
        local current_job = os.getenv('SLURM_ARRAY_JOB_ID')
        if current_job then
          local command = string.format('cd %s && mySubmit -l mem=8,job_dep=%s %s',paths.dirname(aggregationScript), paths.basename(aggregationScript),current_job )
          os.execute(command)
          print(string.format('Executing command:\n%s',command))
        else
          print(string.format('Could not get current job id and therefore cannot launch aggregation script located at:\n%s',aggregationScript))
        end
      end
      
      outFile:write(codeTemplate)
      io.close(outFile)
      print('Created agg scripts at file://' .. aggregationScript)
      _fbd.enter()
    else
      print(string.format("\nAggregation script already found, this job won't write to:\n%s\n",aggregationScript))
    end

  end

  fullState.aggregationScriptWritten = true

end

M.saveForRNGSweep = function(fullState)
	local matio = require 'matio' 
	local output ={}
  --add losses
  --we always have our trainLoss, sometimes we want to add validation or test loss
	output.trainLoss = fullState.trainSetLoss
  if fullState.plotting.losses then
    for k,v in pairs(fullState.plotting.losses) do
      output[k] = fullState[v]
    end
  end

  --add accuracy and the confusion matrices behind them
  for k,v in pairs(fullState.plotting.accuracy) do
    output[v] = fullState[v]
    if fullState[k] and fullState[k].mat then --confusion matrix
      output[k] = fullState[k].mat
    end
  end

	if fullState.args.subj_data.run_single_subj then
		output.subj_id = fullState.data:getSubjID()
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
 end

M.saveForRNGSweepOLD= function(fullState)
	local matio = require 'matio' 
	local output ={}
	output.trainLoss = fullState.trainSetLoss
	output.validLoss = fullState.validSetLoss
	output.trainClassAcc = fullState.trainAvgClassAcc
	output.validClassAcc = fullState.validAvgClassAcc
  output.testClassAcc = fullState.testAvgClassAcc
	--save actual confusion matrix
	output.trainConfMatrix = fullState[M.__getConfusionMatrixName('train')].mat
	output.validConfMatrix = fullState[M.__getConfusionMatrixName('valid')].mat
	output.testConfMatrix = fullState[M.__getConfusionMatrixName('test')].mat
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

	if fullState.testAvgClassAccSubset then
		output.testAvgClassAccSubset = fullState.testAvgClassAccSubset
		output.testConfMatrixSubset = fullState[M.__getConfusionMatrixName('test') .. '_subset'].mat
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
	--print('Final Train Class Acc: '  .. output.trainClassAcc[fullState.trainingIteration])
	--print('Final Valid Class Acc: '  .. output.validClassAcc[fullState.trainingIteration])

end

-- plot helper fns
M.__plotSymbol = function(plotTable, name, values) 
  table.insert(plotTable, {name, values, 'lines lw 4'})
end

M.__makeAndSavePlot = function(saveFile, title, plots)
  require 'gnuplot'
  local pngfig = gnuplot.pngfigure(saveFile)
  gnuplot.raw('set key outside')
  gnuplot.plot(plots)
  gnuplot.raw('set terminal png size 2048,768')
  gnuplot.grid('on')
  gnuplot.raw('set title "' .. title .. '"')
  gnuplot.plotflush()
  gnuplot.close(pngfig)
end

M.__makeAndSaveHist = function(saveFile, title, distribution, bins)
  require 'gnuplot'
  local pngfig = gnuplot.pngfigure(saveFile)
  gnuplot.raw('set terminal png size 1024,768')
  gnuplot.plot(torch.range(1,bins):long(),distribution,'boxes lw 2')
  gnuplot.grid('on')
  gnuplot.title(title)
  gnuplot.plotflush()
  gnuplot.close(pngfig)
end

M.__plotHist = function(fullState, title, distribution, bins, classIdx)
  local newSaveFile, saveFile = '', ''
  if fullState.args.subj_data.run_single_subj then
    newSaveFile = sleep_eeg.utils.insertDirToSaveFile(fullState.args.save_file, fullState.data:getSubjID())
    saveFile = sleep_eeg.utils.replaceTorchSaveWithPngSave(newSaveFile, 'Hist_' .. title .. tostring(classIdx))
  else
    newSaveFile = fullState.args.save_file
    saveFile = sleep_eeg.utils.replaceTorchSaveWithPngSave(fullState.args.save_file, 'Hist_' .. title .. tostring(classIdx))
  end
	print('Saving plot to: ' .. sleep_eeg.utils.fileToURI(saveFile))
    title = fullState.args.driver_name .. ' ' .. title .. ' Max Index Hist: ' .. tostring(classIdx)
	M.__makeAndSaveHist(saveFile, title, distribution, bins)

end

M.plotForRNGSweep = function(fullState)
	local iteration = fullState.trainingIteration
		--make two plots: one for losses, one for classification accuracy
	--loss plots
	local lossPlots = {}

  M.__plotSymbol(lossPlots, 'Train Loss', fullState.trainSetLoss[{{1,iteration}}])
  if fullState.plotting.losses then
    for k,v in pairs(fullState.plotting.losses) do
      local keyName = v
      M.__plotSymbol(lossPlots, k, fullState[keyName][{{1,iteration}}])
    end
  end

  local newSaveFile, saveFile = '', ''
  if fullState.args.subj_data.run_single_subj then
    newSaveFile = sleep_eeg.utils.insertDirToSaveFile(fullState.args.save_file, fullState.data:getSubjID())
    saveFile = sleep_eeg.utils.replaceTorchSaveWithPngSave(newSaveFile, 'Losses')
  else
    newSaveFile = fullState.args.save_file
    saveFile = sleep_eeg.utils.replaceTorchSaveWithPngSave(fullState.args.save_file, 'Losses')
  end
	print('Saving plot to: ' .. sleep_eeg.utils.fileToURI(saveFile))
	local title = fullState.args.driver_name .. '\\nLosses' .. 
	  ' ' .. fullState.args.save_file
	M.__makeAndSavePlot(saveFile, title, lossPlots)
	
	--class acc plots
  if fullState.plotting.accuracy then
    local classAccPlots = {}
    for k,v in pairs(fullState.plotting.accuracy) do
      local keyName = v
      M.__plotSymbol(classAccPlots, k, fullState[keyName][{{1,iteration}}])
    end
    saveFile = sleep_eeg.utils.replaceTorchSaveWithPngSave(newSaveFile, 'ClassAcc')
    print('Saving plot to: ' .. sleep_eeg.utils.fileToURI(saveFile))
    title = fullState.args.driver_name .. '\\nClass Acc' .. 
      ' ' .. fullState.args.save_file
    M.__makeAndSavePlot(saveFile, title, classAccPlots)
  end

end

M.__getConfusionMatrixName = function(trainValidOrTestData, outputTableIndex)
  assert(trainValidOrTestData and type(trainValidOrTestData) == 'string')
  assert(trainValidOrTestData == 'train' or trainValidOrTestData == 'test' or 
    trainValidOrTestData == 'valid', 'Only valid values are "train", ' ..
    ' "valid" or "test"')
	local confMatrixKeyName = trainValidOrTestData 
  if outputTableIndex then
    confMatrixKeyName = confMatrixKeyName .. outputTableIndex .. '_confMatrix'
  else
    confMatrixKeyName = confMatrixKeyName .. '_confMatrix'
  end
	return confMatrixKeyName
end

function M.__fillDistribution(distribution, maxModule)
  local numExamples = maxModule.output:size(1)
  local numClasses = maxModule.output:size(3)
  local maxIdx = 0
  for exampleIdx = 1, numExamples do
    for classIdx = 1, numClasses do
      maxIdx = maxModule.indices[{exampleIdx, 1, classIdx}] + 1 --indices are zero-indexed
      distribution[{classIdx, maxIdx}] = distribution[{classIdx, maxIdx}] + 1
    end
  end
end

M.getDistributionOfMaxTimepoints = function(fullState)
  --max_temp_conv
  local moduleNumber = 3
  --shallow_max_temp_conv
  if torch.type(fullState.network.modules[2]) == 'nn.TemporalMaxPooling' then
    moduleNumber = 2
  elseif torch.type(fullState.network.modules[3]) == 'nn.TemporalMaxPooling' then
    moduleNumber = 3
  elseif torch.type(fullState.network.modules[4]) == 'nn.TemporalMaxPooling' then
    moduleNumber = 4
  elseif torch.type(fullState.network.modules[5]) == 'nn.TemporalMaxPooling' then
    moduleNumber = 5
  end
  assert(torch.type(fullState.network.modules[moduleNumber]) == 'nn.TemporalMaxPooling', "Can only add this hook if we have a temporal max pooling module, which is usually the 3rd module in state.network.  Either you have the wrong network type or the assumption about the max pooling module being the 3rd module is no longer valid.  Either way, check yourself before you wreck yourself.")
  local model = fullState.network
  local maxModule = model.modules[moduleNumber]
  local numTimePoints = fullState.data:size(2)
  local numClasses = fullState.data.num_classes
  local trainDistribution = torch.LongTensor(numClasses, numTimePoints):zero()
  local validDistribution = torch.LongTensor(numClasses, numTimePoints):zero()
  model:evaluate() --make sure we're in evaluation mode

  --training
  model:forward(fullState.data:getTrainData()) --will populate maxModule.output
  M.__fillDistribution(trainDistribution, maxModule)
  for classIdx = 1, numClasses do
    M.__plotHist(fullState, 'Train', trainDistribution[{classIdx,{}}]:view(-1), numTimePoints, classIdx)
  end

  --validation
  model:forward(fullState.data:getValidData()) --will populate maxModule.output
  M.__fillDistribution(validDistribution, maxModule)
  for classIdx = 1, numClasses do
    M.__plotHist(fullState, 'Valid', validDistribution[{classIdx,{}}]:view(-1), numTimePoints, classIdx)
  end
  
end

M.subsetConfusionMatrix = function(fullState, ...)
	assert(fullState and torch.type(fullState) == 'sleep_eeg.State',
		'Must pass in an object of type sleep_eeg.State')
	local args, trainValidOrTestData, allClassNames, subsetClassIdx, outputTableIndex = dok.unpack(
	  {...},
	  'subsetConfusionMatrix',
	  'Makes a hook for a confusion matrix that ignores certain outputs',
	  {arg='trainValidOrTestData', type ='string', help='hook for "train", "valid" or "test" set',req = true},
	  {arg='allClassNames', type ='table', help='table of class names',req = true},
	  {arg='subsetClassIdx', type ='table', help='list-like table of class indexes to keep',req = true},
	  {arg='outputTableIndex', type ='number', help='in the case our network output/targets are table, what index into the table do we want',req = false}
	)
	--this is for the case where we're training a classifier on multiple classes, but 
	--we just want to consider the accuracy for a subset of those classes
	local confMatrixKeyName = M.__getConfusionMatrixName(trainValidOrTestData, outputTableIndex) .. '_subset'

	if not fullState[confMatrixKeyName] then
		fullState:add(confMatrixKeyName, optim.SubsetConfusionMatrix(allClassNames, subsetClassIdx), false)
	end

	M.__updateConfusionMatrix(fullState, trainValidOrTestData, confMatrixKeyName, true, outputTableIndex)

end

M.confusionMatrix = function(fullState, trainValidOrTestData, classNames, outputTableIndex)
	local optim = require 'optim'
	trainValidOrTestData = trainValidOrTestData or 'train' --valid values = 'train', 'test', 'valid'
	local confMatrixKeyName = M.__getConfusionMatrixName(trainValidOrTestData, outputTableIndex)

	if not fullState[confMatrixKeyName] then
		if classNames then
			fullState:add(confMatrixKeyName, optim.ConfusionMatrix(classNames), false)
		else
			error('This should never get here - we can fix this later')
			fullState[confMatrixKeyName] = optim.ConfusionMatrix()
		end
	end

	M.__updateConfusionMatrix(fullState, trainValidOrTestData, confMatrixKeyName, false, outputTableIndex)
end

M.__updateConfusionMatrix = function(fullState, trainValidOrTestData, confMatrixKeyName, isSubset, outputTableIndex)
	local suffix = ''
	if isSubset then 
		suffix = "Subset"
	end

  local getOutput = function(data)
    if not outputTableIndex then
      return data
    else
      return data[outputTableIndex]
    end
  end

  local getData = function(trainValidOrTestData)
    if trainValidOrTestData == 'train' then
      return fullState.data:getTrainData()
    elseif trainValidOrTestData == 'test' then
      return fullState.data:getTestData()
    elseif trainValidOrTestData == 'valid' then
      return fullState.data:getValidData()
    end
  end
  local getTargets = function(trainValidOrTestData)
    if trainValidOrTestData == 'train' then
      return fullState.data:getTrainTargets()
    elseif trainValidOrTestData == 'test' then
      return fullState.data:getTestTargets()
    elseif trainValidOrTestData == 'valid' then
      return fullState.data:getValidTargets()
    end
  end

  --where we list fields to plot
  if not fullState.plotting then
    fullState.plotting = {}
    if not fullState.plotting.accuracy then
      fullState.plotting.accuracy = {}
    end
  end

  local outputIndexString = ''
  if outputTableIndex then
    outputIndexString = tostring(outputTableIndex)
  end
	--here we actually look into fullState and get it's output, we're breaking generality
	--here just to get this done
	fullState[confMatrixKeyName]:zero()

  local classAccKey = trainValidOrTestData .. 'AvgClassAcc' .. outputIndexString .. suffix
  if not fullState[classAccKey] then
    fullState:add(classAccKey, torch.FloatTensor(fullState.args.training.maxTrainingIterations):fill(-1.0), true)
  end
  if not fullState.plotting.accuracy[confMatrixKeyName] then
    fullState.plotting.accuracy[confMatrixKeyName] = classAccKey
  end

  local data = getData(trainValidOrTestData)
  local targets = getTargets(trainValidOrTestData)
  local numExamples = data:size(1)
  local shuffle = torch.randperm(numExamples):long()
  local numMiniBatches = sleep_eeg.utils.getNumMiniBatches(numExamples, fullState.args.miniBatchSize)

  for miniBatchIdx = 1, numMiniBatches do

    local miniBatchTrials = sleep_eeg.utils.getMiniBatchTrials(shuffle, miniBatchIdx, fullState.args.miniBatchSize)

    local modelOut = getOutput(fullState.network:forward(data:index(1,miniBatchTrials)))
    local batch_targets = getOutput(sleep_eeg.utils.indexIntoTensorOrTableOfTensors(targets,1,miniBatchTrials) )

    --add outputs to conf matrix
    fullState[confMatrixKeyName]:batchAdd(modelOut, batch_targets)

  end
  fullState[confMatrixKeyName]:updateValids() --update confMatrix
  fullState[classAccKey][fullState.trainingIteration] = fullState[confMatrixKeyName].totalValid

  if fullState.trainingIteration % M.OUTPUT_EVERY_X_ITERATIONS == 0 then
    print(classAccKey .. trainValidOrTestData .. ' accuracy ' .. (outputTableIndex and outputTableIndex or '' ) ..  ':' .. fullState[confMatrixKeyName].totalValid)
  end

end


M.saveNetwork = function(fullState)
  local netFileOut = ''
  if fullState.args.subj_data and fullState.args.subj_data.run_single_subj then
    local newSaveFile = sleep_eeg.utils.insertDirToSaveFile(fullState.args.save_file, fullState.data:getSubjID())
	netFileOut = sleep_eeg.utils.replaceTorchSaveWithNetSave(newSaveFile)
  else
	netFileOut = sleep_eeg.utils.replaceTorchSaveWithNetSave(fullState.args.save_file)
  end

  if not fullState.args.subj_data.predict_subj then
	  local net = fullState.network:clone()
	  sleep_eeg.utils.ghettoClearStateSequential(net)
	  torch.save(netFileOut, {net = net, trainingIteration = fullState.trainingIteration} )
  	  print('Saved network to: ' .. netFileOut)
  else
	torch.save(netFileOut, {net = fullState.network, trainingIteration = fullState.trainingIteration} )
  	print('Saved network to: ' .. netFileOut)
  end
end

return M
