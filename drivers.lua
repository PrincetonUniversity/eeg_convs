
local M = {}

--these are settings that we want to share across different drivers, jhust makes it easier
local SHARED_SETTINGS = {}
SHARED_SETTINGS.maxTrainingIterations = 10000

M.train = function(fullState)
	assert(torch.type(fullState) == 'sleep_eeg.State')
	local optimizers = sleep_eeg.optimizers

	local options = fullState.args.training --shortcut to our args

	--TODO: We actually do want to save the state of our optimizer, b/c some
	--optimizers have their own internal state
	if not fullState.optimizer and not fullState.optimSettings then
		--make our optimizer
		local optim, optimSettings = optimizers.getOptimizer(options.optimName, options.learningRate)
		fullState:add('optimizer', optim,false)
		fullState:add('optimSettings', optimSettings, false)
	elseif utils.nilXOR(fullState.network, fullState.criterion) then
		error([[You've managed to save one, but NOT BOTH, of the following values:
				- state.optim
				- state.optimSettings
				We can't load just one because optim depends on optimSettings and 
				it's not easy to check if they match up']])
	end

	if not fullState.trainingIteration then
		fullState:add('trainingIteration',0,true)
	end

	--actually run the optimizer
	local shouldTerminateEarly = false
	local start = torch.tic()
  print('Starting to train...')
	while fullState.trainingIteration < options.maxTrainingIterations 
		and (not shouldTerminateEarly) do

		fullState.trainingIteration = fullState.trainingIteration + 1

		optimizers.performTrainIteration(fullState)

		if #options.trainingIterationHooks > 0 then
			for hookIdx = 1, #options.trainingIterationHooks do
				options.trainingIterationHooks[hookIdx](fullState)
			end
		end

		if options.earlyTerminationFn then
			shouldTerminateEarly = options.earlyTerminationFn(fullState)
			if shouldTerminateEarly then
				print('Terminating early!')
			end
		end

		--garbage collect every 100 training iterations
		if fullState.trainingIteration % 10 == 0 then
			print('10 iterations took: ' .. torch.toc(start) .. 'secs')
			start = torch.tic()
			collectgarbage()
		end
	end

	--finally, if we have any hooks to eecute after completion
	if #options.trainingCompleteHooks > 0 then
		for hookIdx = 1, #options.trainingCompleteHooks do
			options.trainingCompleteHooks[hookIdx](fullState)
		end
	end
end

M.fullConvWake = function()
	local utils = sleep_eeg.utils

	--all arguments we will ever need to run this function
	--create sim data
	args = {}
	args.driver_name = 'fullConvWake'
	--TODO: rng state is NOT saved right now
	args.rng_seed = '102387'

  --subj data arguments
  args.subj_data = {}
  args.subj_data.filename = './torch_exports/wake_ERP_cuelocked_all_4ms.mat'
  args.subj_data.percent_train = 65
  args.subj_data.percent_valid = 20
  args.subj_data.do_split_loso = false
  args.subj_data.run_single_subj = true

  --let's populate any job specific args we're sweeping over, because we need to get
  --subject_idx before we can populate subj_data
  sleep_eeg.utils.populateArgsBasedOnJobNumber(args)

  --with the subj_data args specified, we go ahead and load the subject data 
  --because other argument values for the network and confusion matix depend on 
  --values that get loaded by subj_data
  local subj_data 
  if args.subj_data.run_single_subj then
	subj_data = sleep_eeg.SingleSubjData(args.subj_data.filename,
		args.subj_data.subj_idx, args.subj_data.percent_valid, 
    	args.subj_data.percent_train)
  else
	subj_data = sleep_eeg.CVBySubjData(args.subj_data.filename, 
    	args.subj_data.do_split_loso, args.subj_data.percent_valid, 
    	args.subj_data.percent_train)
  end
	
	--network args
	args.network = {}
	args.network.numHiddenUnits = subj_data.num_classes
	args.network.numHiddenLayers = 1
	args.network.num_output_classes = subj_data.num_classes
	--training args, used by sleep_eeg.drivers.train()
	args.training = {}
	args.training.optimName = 'adam'
	args.training.learningRate = .001
	args.training.maxTrainingIterations =  SHARED_SETTINGS.maxTrainingIterations
	args.training.trainingIterationHooks = {} -- populated below
	args.training.earlyTerminationFn = nil --populated below just put this here so that, all args are easy to see
	args.training.trainingCompleteHooks = {}

	------------------------------------------------------------------------
	--populate hooks
	
	--TODO: here we define our training iteration hooks, completion hooks, etc
	--make a closure that will pass in the 'training' arg to our 
	local trainConfMatrix = function(state)
		sleep_eeg.hooks.confusionMatrix(state, 'train', subj_data.classnames )
	end
	args.training.trainingIterationHooks[1] = trainConfMatrix

	--make a closure that will pass in the 'valid' arg to our 
	local validConfMatrix = function(state)
		sleep_eeg.hooks.confusionMatrix(state, 'valid', subj_data.classnames)
	end
	args.training.trainingIterationHooks[2] = validConfMatrix
	args.training.trainingIterationHooks[3] = sleep_eeg.hooks.validLoss

	--make a closure that will pass in the 'train' arg to a "subsetConfusionMatrix"
	--which only cares about performance on a subset of all possible classes
	local trainConfSubsetMatrix = function(state)
		sleep_eeg.hooks.subsetConfusionMatrix(state, 'train', subj_data.classnames, {1,2})--only do faces and places
	end
	args.training.trainingIterationHooks[3] = trainConfSubsetMatrix

	--make a closure that will pass in the 'valid' arg to subsetConfusionMatrix
	local validConfSubsetMatrix = function(state)
		sleep_eeg.hooks.subsetConfusionMatrix(state, 'valid', subj_data.classnames, {1,2})
	end
	args.training.trainingIterationHooks[4] = validConfSubsetMatrix


	--Training Completed Hooks
	args.training.trainingCompleteHooks[1] = function(state)
		return sleep_eeg.hooks.saveForRNGSweep(state)
	end

	if args.subj_data.run_single_subj then
		args.training.trainingCompleteHooks[2] = sleep_eeg.hooks.plotForRNGSweep
	end

	--make a closure for our early termination fn
	args.training.earlyTerminationFn = function(state)
		return sleep_eeg.terminators.trainAndValidAvgClassAccuracyHigh(state,0.6)
	end

	args.save_file = utils.saveFileNameFromDriversArgs(args,args.driver_name)
	--end args definition 
	------------------------------------------------------------------------

	--this will get reload state if args.save_file already exists
	--otherwise, just keeps saving there
	state = sleep_eeg.State(args.save_file)

	--set random seed
	if not state.rngState then
		torch.manualSeed(args.rng_seed)
	else
		torch.setRNGState(state.rngState)
	end

	if not state.args then
		state:add('args', args, true)
	end

	if not state.data then
		state:add('data',subj_data,false)
	end

	--create our network and criterion  (network type determines criterion type)
	if not state.network and not state.criterion then
    print('making network started...')
    print(args.network)
		local network, criterion = 
			sleep_eeg.models.createMaxTempConvClassificationNetwork( 
				state.data:getTrainData(), args.network.numHiddenUnits, 
				args.network.numHiddenLayers, state.data.num_classes)
			--sleep_eeg.models.createNoMaxTempConvClassificationNetwork( 
				--state.data:getTrainData(), args.network.numHiddenUnits, 
				--args.network.numHiddenLayers, state.data.num_output_classes)

    print('making network finished...')
		state:add('network',network, true)
		state:add('criterion',criterion, true)
	elseif utils.nilXOR(state.network, state.criterion) then
		error([[You've managed to save one, but NOT BOTH, of the following values:
				- state.network
				- state.criterion
				We can't load just one because making the network determines the type of 
				criterion you need.]])
	end

	--we load params and gradParams together
	if not state.params and not state.gradParams then
		local params, gradParams = state.network:getParameters()
		state:add('params',params,true)
		state:add('gradParams',gradParams,true)
	elseif utils.nilXOR(state.params, state.gradParams) then
		error([[You've managed to save one, but NOT BOTH, of the following values:
				- state.params
				- state.gradParams
				We can't load just one because network:getParameters() manipulates both. 
				And you don't really want to call network:getParameters() on the same 
				network twice cause spooky things happen']])
	end

	--little output about our network
	print('-------------------------------------------------------------------')
	print('Network information: ')
	print(state.network)
	print('With a total of ' ..state.params:numel() .. ' parameters')

	--finally call the optimizer
	M.train(state)

end

M.fullConv = function()
	local utils = sleep_eeg.utils

	--all arguments we will ever need to run this function
	--create sim data
	args = {}
	args.driver_name = 'fullConvADAM'
	--TODO: rng state is NOT saved right now
	args.rng_seed = '102387'

  --subj data arguments
  args.subj_data = {}
  args.subj_data.isSim = true
  if args.subj_data.isSim then
  	args.subj_data.filename = './torch_exports/sleep_ERP_cuelocked_all_4ms.mat'
  else
	args.subj_data.filename = './torch_exports/sleep_ERP_cuelocked_all_4ms_sim.mat'
  end
  args.subj_data.percent_train = 65
  args.subj_data.percent_valid = 20
  args.subj_data.do_split_loso = false
  args.subj_data.run_single_subj = true

  --let's populate any job specific args we're sweeping over, because we need to get
  --subject_idx before we can populate subj_data
  sleep_eeg.utils.populateArgsBasedOnJobNumber(args)

  --with the subj_data args specified, we go ahead and load the subject data 
  --because other argument values for the network and confusion matix depend on 
  --values that get loaded by subj_data
  local subj_data 
  if args.subj_data.run_single_subj then
	  subj_data = sleep_eeg.SingleSubjData(args.subj_data.filename,
		  args.subj_data.subj_idx, args.subj_data.percent_valid, 
		  args.subj_data.percent_train)
  else
	  subj_data = sleep_eeg.CVBySubjData(args.subj_data.filename, 
		  args.subj_data.do_split_loso, args.subj_data.percent_valid, 
		  args.subj_data.percent_train)
  end

  --network args
  args.network = {}
  args.network.numHiddenUnits = subj_data.num_classes
  args.network.numHiddenLayers = 1
  args.network.num_output_classes = subj_data.num_classes
  --training args, used by sleep_eeg.drivers.train()
  args.training = {}
  args.training.optimName = 'adam'
  args.training.learningRate = .00001
  args.training.maxTrainingIterations =  SHARED_SETTINGS.maxTrainingIterations
  args.training.trainingIterationHooks = {} -- populated below
  args.training.earlyTerminationFn = nil --populated below just put this here so that, all args are easy to see
  args.training.trainingCompleteHooks = {}

  ------------------------------------------------------------------------
  --populate hooks

  --TODO: here we define our training iteration hooks, completion hooks, etc
  --make a closure that will pass in the 'training' arg to our 
  local trainConfMatrix = function(state)
	  sleep_eeg.hooks.confusionMatrix(state, 'train', subj_data.classnames )
  end
  args.training.trainingIterationHooks[1] = trainConfMatrix
  --make a closure that will pass in the 'valid' arg to our 
  local validConfMatrix = function(state)
	  sleep_eeg.hooks.confusionMatrix(state, 'valid', subj_data.classnames)
  end
  args.training.trainingIterationHooks[2] = validConfMatrix
  args.training.trainingIterationHooks[3] = sleep_eeg.hooks.validLoss
  args.training.trainingIterationHooks[4] = sleep_eeg.hooks.logWeightToUpdateNormRatio

  --Training Completed Hooks
  args.training.trainingCompleteHooks[1] = function(state)
	  return sleep_eeg.hooks.randomClassAcc(state, subj_data.num_classes)
  end

  args.training.trainingCompleteHooks[2] = function(state)
	  return sleep_eeg.hooks.saveForRNGSweep(state)
  end

  --we really only want to plot automatically if we're doing subjects separately
  if args.subj_data.run_single_subj then
  	args.training.trainingCompleteHooks[3] = sleep_eeg.hooks.plotForRNGSweep
  end

  --make a closure for our early termination fn
  if not args.subj_data.isSim then
	  args.training.earlyTerminationFn = function(state)
		  return sleep_eeg.terminators.trainAndValidAvgClassAccuracyHigh(state,0.7)
	  end
  end

  args.save_file = utils.saveFileNameFromDriversArgs(args,args.driver_name)
  --end args definition 
  ------------------------------------------------------------------------

  --this will get reload state if args.save_file already exists
  --otherwise, just keeps saving there
  state = sleep_eeg.State(args.save_file)

  --set random seed
  if not state.rngState then
	  torch.manualSeed(args.rng_seed)
  else
	  torch.setRNGState(state.rngState)
  end

  if not state.args then
	  state:add('args', args, true)
  end

  if not state.data then
	  state:add('data',subj_data,false)
  end

  --create our network and criterion  (network type determines criterion type)
  if not state.network and not state.criterion then
	  print('making network started...')
	  print(args.network)
	  local network, criterion = 
	  sleep_eeg.models.createMaxTempConvClassificationNetwork( 
		  state.data:getTrainData(), args.network.numHiddenUnits, 
		  args.network.numHiddenLayers, state.data.num_classes)
	  --sleep_eeg.models.createNoMaxTempConvClassificationNetwork( 
	  --state.data:getTrainData(), args.network.numHiddenUnits, 
	  --args.network.numHiddenLayers, state.data.num_output_classes)

	  print('making network finished...')
	  state:add('network',network, true)
	  state:add('criterion',criterion, true)
  elseif utils.nilXOR(state.network, state.criterion) then
	  error([[You've managed to save one, but NOT BOTH, of the following values:
				- state.network
				- state.criterion
				We can't load just one because making the network determines the type of 
				criterion you need.]])
  end

  --we load params and gradParams together
  if not state.params and not state.gradParams then
	  local params, gradParams = state.network:getParameters()
	  state:add('params',params,true)
	  state:add('gradParams',gradParams,true)
  elseif utils.nilXOR(state.params, state.gradParams) then
	  error([[You've managed to save one, but NOT BOTH, of the following values:
				- state.params
				- state.gradParams
				We can't load just one because network:getParameters() manipulates both. 
				And you don't really want to call network:getParameters() on the same 
				network twice cause spooky things happen']])
  end

  --little output about our network
  print('-------------------------------------------------------------------')
  print('Network information: ')
  print(state.network)
  print('With a total of ' ..state.params:numel() .. ' parameters')

  --finally call the optimizer
  M.train(state)

end

M.noMaxOut = function()
	local utils = sleep_eeg.utils

	--all arguments we will ever need to run this function
	--create sim data
	args = {}
	args.driver_name = 'noMaxOut'
	--TODO: rng state is NOT saved right now
	args.rng_seed = '102387'

	--subj data arguments
	args.subj_data = {}
	args.subj_data.filename = './torch_exports/sleep_ERP_cuelocked_all_4ms.mat'
	args.subj_data.percent_train = 65
	args.subj_data.percent_valid = 20
	args.subj_data.do_split_loso = false
	--with the subj_data args specified, we go ahead and load the subject data 
	--because other argument values for the network and confusion matix depend on 
	--values that get loaded by subj_data
	local subj_data = sleep_eeg.CVBySubjData(args.subj_data.filename, 
		args.subj_data.do_split_loso, args.subj_data.percent_valid, 
		args.subj_data.percent_train)

	--network args
	args.network = {}
	args.network.numHiddenUnits = subj_data.num_classes
	args.network.numHiddenLayers = 1
	args.network.num_output_classes = subj_data.num_classes
	--training args, used by sleep_eeg.drivers.train()
	args.training = {}
	args.training.optimName = 'adam'
	args.training.learningRate = .001
	args.training.maxTrainingIterations =  SHARED_SETTINGS.maxTrainingIterations
	args.training.trainingIterationHooks = {} -- populated below
	args.training.earlyTerminationFn = nil --populated below just put this here so that, all args are easy to see
	args.training.trainingCompleteHooks = {}

	------------------------------------------------------------------------
	--populate hooks

	--TODO: here we define our training iteration hooks, completion hooks, etc
	--make a closure that will pass in the 'training' arg to our 
	local trainConfMatrix = function(state)
		sleep_eeg.hooks.confusionMatrix(state, 'train', subj_data.classnames )
	end
	args.training.trainingIterationHooks[1] = trainConfMatrix
	--make a closure that will pass in the 'valid' arg to our 
	local validConfMatrix = function(state)
		sleep_eeg.hooks.confusionMatrix(state, 'valid', subj_data.classnames)
	end
	args.training.trainingIterationHooks[2] = validConfMatrix

	--Training Completed Hooks
	args.training.trainingCompleteHooks[1] = function(state)
		return sleep_eeg.hooks.saveForRNGSweep(state)
	end

	--make a closure for our early termination fn
	args.training.earlyTerminationFn = function(state)
		return sleep_eeg.terminators.trainAndValidAvgClassAccuracyHigh(state,0.7)
	end

	--lastly, let's populate any job specific args we're sweeping over
	sleep_eeg.utils.populateArgsBasedOnJobNumber(args)
	args.save_file = utils.saveFileNameFromDriversArgs(args,args.driver_name)
	--end args definition 
	------------------------------------------------------------------------

	--this will get reload state if args.save_file already exists
	--otherwise, just keeps saving there
	state = sleep_eeg.State(args.save_file)

	--set random seed
	if not state.rngState then
		torch.manualSeed(args.rng_seed)
	else
		torch.setRNGState(state.rngState)
	end

	if not state.args then
		state:add('args', args, true)
	end

	if not state.data then
		state:add('data',subj_data,false)
	end

	--create our network and criterion  (network type determines criterion type)
	if not state.network and not state.criterion then
		print('making network started...')
		print(args.network)
		local network, criterion = 
		sleep_eeg.models.createNoMaxTempConvClassificationNetwork( 
			state.data:getTrainData(), args.network.numHiddenUnits, 
			args.network.numHiddenLayers, state.data.num_output_classes)

		print('making network finished...')
		state:add('network',network, true)
		state:add('criterion',criterion, true)
	elseif utils.nilXOR(state.network, state.criterion) then
		error([[You've managed to save one, but NOT BOTH, of the following values:
				- state.network
				- state.criterion
				We can't load just one because making the network determines the type of 
				criterion you need.]])
	end

	--we load params and gradParams together
	if not state.params and not state.gradParams then
		local params, gradParams = state.network:getParameters()
		state:add('params',params,true)
		state:add('gradParams',gradParams,true)
	elseif utils.nilXOR(state.params, state.gradParams) then
		error([[You've managed to save one, but NOT BOTH, of the following values:
				- state.params
				- state.gradParams
				We can't load just one because network:getParameters() manipulates both. 
				And you don't really want to call network:getParameters() on the same 
				network twice cause spooky things happen']])
	end

	--little output about our network
	print('-------------------------------------------------------------------')
	print('Network information: ')
	print(state.network)
	print('With a total of ' ..state.params:numel() .. ' parameters')

	--finally call the optimizer
	M.train(state)

end

return M
