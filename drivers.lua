
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

	  fullState.network:training() --added for dropout functionality
      optimizers.performTrainIteration(fullState)
	  fullState.network:evaluate()

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

local makeConfigName = function(args, cmdOptions)

  local snake_to_CamelCase = function (s)
    return s:gsub("_%w", function (u) return u:sub(2,2):upper() end)
  end
  local function firstToUpper(s)
    return s:gsub("^%l", string.upper)
  end

  local name = snake_to_CamelCase(cmdOptions.network_type) .. firstToUpper(cmdOptions.optim)
  if cmdOptions.dropout_prob > 0 then
  	name = name .. 'Drop' .. tostring(cmdOptions.dropout_prob)
  end
  --simulated data indicator
  if cmdOptions.simulated >= 0 then
    simString = 'Sim' .. tostring(cmdOptions.simulated)
    name = name .. simString
  end
  if cmdOptions.wake then
    name = name .. 'Wake'
  else
    name = name .. 'Sleep'
  end
  --per subject indicator
  if cmdOptions.run_single_subj then 
    name = name .. 'PerSubj'
  end
  return name
end

local initArgs = function()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Neural Networks for EEG')
  cmd:text()
  cmd:text('Options')
  cmd:option('-simulated', -1, '-1 = no sim data, 0 = basic, 1 = noisy, 2 = temp conv')
  cmd:option('-percent_train', 65, 'percent of data to use for training')
  cmd:option('-percent_valid', 20, 'percent of data to use for validation')
  cmd:option('-loso',false, 'leave-one-subject-out validation? NOTE: currently not implemented')
  cmd:option('-run_single_subj',false, 'run within subject analysis')
  cmd:option('-wake',false, 'if false, run sleep else run wake')
  cmd:option('-optim','adam', 'optimizer to use, supported optimizers = "sgd" or "adam"')
  cmd:option('-learning_rate', 1e-5, 'learning rate for optimizer')
  cmd:option('-max_iterations', 20000, 'max number of iterations to optimize for (can still terminate early)')
  cmd:option('-early_termination', -1, '-1 = no early termination, values between 0 and 1 will terminate optimization if training and validation classification accuracy exceed this value')
  cmd:option('-network_type', 'max_temp_conv', 'network type to use, valid values = "max_temp_conv", "no_max_temp_conv", and "fully_connected"')
  cmd:option('-dropout_prob', -1, 'Probability of input dropout.')
  cmd:option('-config_name', '', 'what we want to call this configuration of arguments; dictates the name of the folder we save data to. leaving this empty will generate directory name based on arguments passed.')
  cmd:option('-subj_index', 0, 'subject index, not ID. only valid for run_single_subj = true')
  cmd:text()
  opt = cmd:parse(arg)
  return opt, cmd
end

M.generalDriver = function()
  --all cmd-line options:
  -- args.network_type 'max_temp_conv', 'no_max_temp_conv', 'fully_connected'
  local utils = sleep_eeg.utils
  local cmdOptions, cmdLine = initArgs()

  --all arguments we will ever need to run this function
  --create sim data
  args = {}
  args.subj_data = {}

  args.rng_seed = '102387'--TODO: rng state is NOT saved right now

  --subj data arguments
  args.subj_data.isSim = cmdOptions.simulated >= 0
  args.subj_data.percent_train = cmdOptions.percent_train
  args.subj_data.percent_valid = cmdOptions.percent_valid
  args.subj_data.do_split_loso = cmdOptions.loso
  args.subj_data.run_single_subj = cmdOptions.run_single_subj
  args.subj_data.wake = cmdOptions.wake
  local fileName = ''
  if cmdOptions.wake then
    fileNameRoot = 'wake_ERP_cuelocked_all_4ms'
  else
    fileNameRoot = 'sleep_ERP_cuelocked_all_4ms'
  end
  if args.subj_data.isSim then
    args.subj_data.filename = './torch_exports/' .. fileNameRoot .. '_sim.mat'
  else
    args.subj_data.filename = './torch_exports/' .. fileNameRoot .. '.mat'
  end

  --let's populate any job specific args we're sweeping over, because we need to get
  --subject_idx before we can populate subj_data
  sleep_eeg.utils.populateArgsBasedOnJobNumber(args)

  if cmdOptions.rng_seed then
    args.rng_seed = cmdOptions.rng_seed
  end
  if cmdOptions.subj_index ~= 0 then
    args.subj_data.subj_idx = cmdOptions.subj_index
  end

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
  args.network.dropout_prob = cmdOptions.dropout_prob
  --training args, used by sleep_eeg.drivers.train()
  args.training = {}
  args.training.optimName = cmdOptions.optim
  args.training.learningRate = cmdOptions.learning_rate
  args.training.maxTrainingIterations =  cmdOptions.max_iterations
  args.training.trainingIterationHooks = {} -- populated below
  args.training.earlyTerminationFn = nil --populated below just put this here so that, all args are easy to see
  args.training.trainingCompleteHooks = {}

  if cmdOptions.config_name == '' then
    args.driver_name = makeConfigName(args,cmdOptions) --if no config_name specified, make from args
  else
    args.driver_name = cmdOptions.config_name
  end

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

  if args.subj_data.wake then
    --make a closure that will pass in the 'train' arg to a "subsetConfusionMatrix"
    --which only cares about performance on a subset of all possible classes
    local trainConfSubsetMatrix = function(state)
      sleep_eeg.hooks.subsetConfusionMatrix(state, 'train', subj_data.classnames, {1,2})--only do faces and places
    end
    args.training.trainingIterationHooks[5] = trainConfSubsetMatrix

    --make a closure that will pass in the 'valid' arg to subsetConfusionMatrix
    local validConfSubsetMatrix = function(state)
      sleep_eeg.hooks.subsetConfusionMatrix(state, 'valid', subj_data.classnames, {1,2})
    end
    args.training.trainingIterationHooks[6] = validConfSubsetMatrix
  end

  --Training Completed Hooks
  args.training.trainingCompleteHooks[1] = function(state)
    return sleep_eeg.hooks.randomClassAcc(state, subj_data.num_classes)
  end

  args.training.trainingCompleteHooks[2] = function(state)
    return sleep_eeg.hooks.saveForRNGSweep(state)
  end

  args.training.trainingCompleteHooks[3] = sleep_eeg.hooks.plotForRNGSweep

  if cmdOptions.network_type == 'max_temp_conv' then 
    args.training.trainingCompleteHooks[4] = sleep_eeg.hooks.getDistributionOfMaxTimepoints
  end

  --make a closure for our early termination fn
  if cmdOptions.early_termination > 0 and cmdOptions.early_termination <= 1 then
    args.training.earlyTerminationFn = function(state)
      return sleep_eeg.terminators.trainAndValidAvgClassAccuracyHigh(state, cmdOptions.early_termination)
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
    local network, criterion = {},{}
    if cmdOptions.network_type == 'max_temp_conv' then 
      network, criterion = sleep_eeg.models.createMaxTempConvClassificationNetwork( 
        state.data:getTrainData(), args.network.numHiddenUnits, 
        args.network.numHiddenLayers, state.data.num_classes, 
		args.network.dropout_prob)
    elseif cmdOptions.network_type == 'no_max_temp_conv' then
      network, criterion = sleep_eeg.models.createNoMaxTempConvClassificationNetwork( 
        state.data:getTrainData(), args.network.numHiddenUnits, 
        args.network.numHiddenLayers, state.data.num_classes, 
		args.network.dropout_prob)
    elseif cmdOptions.network_type == 'fully_connected' then
      network, criterion = sleep_eeg.models.createFullyConnectedNetwork(
	  	state.data:getTrainData(), args.network.numHiddenUnits, 
		args.network.numHiddenLayers, state.data.num_classes, 
		args.network.dropout_prob)
    end
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
