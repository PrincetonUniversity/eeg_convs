require 'paths'
local M = {}

M.fileExists = function(fname)
	local file = io.open(fname,'r')
	if file ~= nil then
		io.close(file)
		return true
	else
		return false
	end
end

M.nilXOR = function(obj1,obj2)
	--returns true if exactly one of the objects is not nil
	return not (obj1) == not not obj2
end

local test_nilXOR = function()
	assert(not M.nilXOR(1,1))
	assert(not M.nilXOR(nil,nil))
	assert(M.nilXOR(nil,1))
	assert(M.nilXOR(1,nil))
end
test_nilXOR()


M.normalizeData = function(data,mean_, std_)
  local numExamples = data:size(1)
  local std = std_ or data:std(1)
  local mean = mean_ or data:mean(1)

  -- actually normalize here
  for i = 1, numExamples do
     data[i]:add(-mean)
     data[i]:cdiv(std)
  end

  return mean, std
end


M.populateArgsBasedOnJobNumber = function(args)
	require 'os'
	local gridOptions = {
		rng_seed = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40},
	}
  --if we want to run a single subject, then we also have to specify which
  --subj_idx we want to run for this job
  if args.subj_data.run_single_subj then 
    --we have 33 subjects
    gridOptions.subj_idx = torch.linspace(1,33,33):totable()
  end

	local job_number = os.getenv('SLURM_ARRAY_TASK_ID')
	if not job_number then
		job_number = 0
	else
		job_number = tonumber(job_number)-1
	end
	local options = sleep_eeg.param_sweep.grid({id = job_number},gridOptions)
	args.rng_seed = options.rng_seed
  if args.subj_data.run_single_subj then
	  args.subj_data.subj_idx = options.subj_idx
  end
end

M.saveFileNameFromDriversArgs = function(args,base_name)
	--build file path
	local driverPrefix = base_name 
	local gitCommitHash = M.getGitCommitNumAndHash()
	local rngSeedString = 'rng_' .. args.rng_seed 
  local learningRateString = 'learnRate_' .. string.format("%.0e",args.training.learningRate)
	local fullPath = paths.concat(dotrc.save_dir,driverPrefix, gitCommitHash)
	if not paths.dir(fullPath) then
		paths.mkdir(fullPath)
	end

	--build filename
	local filename = learningRateString .. '_' .. rngSeedString .. '.th7'

	local fullFilename = paths.concat(fullPath,filename)
	print(fullFilename)
	return  fullFilename
end


M.copyArgsRemoveUnsupportedMatioTypes = function (source)
	local tablex = require 'pl.tablex'
	local target = tablex.deepcopy(source)
	target.training.trainingIterationHooks = nil
	target.training.earlyTerminationFn = nil
	target.training.trainingCompleteHooks = nil
	return target
end




M.replaceTorchSaveWithNetSave = function(torchFilename, suffix)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	if suffix then
		return paths.concat(dir,baseFilename .. suffix .. '.net')
	else
		return paths.concat(dir,baseFilename .. '.net')
	end
end


M.replaceTorchSaveWithMatSave = function(torchFilename)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	return paths.concat(dir,baseFilename .. '.mat')
end

M.replaceTorchSaveWithEpsSave = function(torchFilename, suffix)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	if suffix then
		return paths.concat(dir,baseFilename .. suffix .. '.eps')
	else
		return paths.concat(dir,baseFilename .. '.eps')
	end
end

M.replaceTorchSaveWithPngSave = function(torchFilename, suffix)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	if suffix then
		return paths.concat(dir,baseFilename .. suffix .. '.png')
	else
		return paths.concat(dir,baseFilename .. '.png')
	end
end

M.insertDirToSaveFile = function(torchFilename, dirToAdd)
	local dir = paths.concat(paths.dirname(torchFilename), dirToAdd)
	local filename = paths.basename(torchFilename)
	if not paths.dir(dir) then
		paths.mkdir(dir)
	end
	return paths.concat(dir, filename)
end

M.matioHelper = function(filename, varNames)
  local matio = require 'matio'
  matio.use_lua_strings = true
  local loaded = {}
  for _, var in pairs(varNames) do
    loaded[var] = matio.load(filename,var)
  end
  return loaded
end

M.getUniqueStrings = function(strTable)
  local numUniqueStrings = 0
  -- keys are unique strings, values are which key number they are
  local uniqueStringIDs = {} 
  --list of which uniqueStringIDs each example belongs to
  local uniqueStringIdxs = {}
  --an ordered list of unique strings
  local uniqueStrings = {}
  --num strings: 
  local size 
  if torch.type(strTable) == 'table' then
    size = #strTable
  elseif string.find(torch.type(strTable),'Tensor') then
    assert(strTable:nDimension() == 1, 
        'this function only works with tables or tensors with one dimension')
    size = strTable:size(1)
  end

  for strIdx = 1, size do
    local currentStr = strTable[strIdx]
    if uniqueStringIDs[currentStr] == nil then -- new string
      numUniqueStrings = numUniqueStrings + 1
      uniqueStringIDs[currentStr] = numUniqueStrings 
      uniqueStrings[numUniqueStrings] = currentStr
    end
    uniqueStringIdxs[strIdx] = uniqueStringIDs[currentStr]
  end
  return uniqueStrings, uniqueStringIdxs
end

--applies to a tensor or a table of tensors
M.getDataFromTableOrTensor = function(data,idxs)
  local canCopyInBulk = false
  if idxs:max()-idxs:min()+1 == idxs:numel() then
    canCopyInBulk = true
  end

  local newData = {}
  local shouldUntableData = false
  if torch.type(data) ~= 'table' then
    data = {data}
    shouldUntableData = true
  end

  for dataIdx = 1, #data do
    if canCopyInBulk then
      newData[dataIdx] = data[dataIdx][{{idxs:min(), idxs:max()}}]
    else
      local size = data[dataIdx]:size():totable()
      size[1] = idxs:numel()
      newData[dataIdx] = 
          torch.Tensor():typeAs(data[dataIdx]):resize(unpack(size))
      for exampleIdx = 1, idxs:numel() do
        newData[dataIdx][exampleIdx] = data[dataIdx][idxs[exampleIdx]]
      end
    end
  end

  if shouldUntableData then
    return newData[1]
  end

  return newData
end

M.splitDataBasedOnLabels = function(data, labels)
  --[
  -- labels can be either a tensor of unique numbers or a table of unique 
  -- elements
  --]
  assert(data:size(1) == labels:size(1))--one label per datapoint
  assert(data:nDimension() == 2)
  local numExamples = data:size(1)
  local numFeatures = data:size(2)

  local uniqueIDs, uniqueIDXs = M.getUniqueStrings(labels)
  uniqueIDXs = torch.Tensor{uniqueIDXs}

  local numClasses = #uniqueIDs
  local maxExamples = 0
  for classIdx = 1, numClasses do
    local numClassExamples = uniqueIDXs:eq(classIdx):sum()
    assert(numClassExamples > 0, 'something fishy going on here')
    if numClassExamples > maxExamples then
      maxExamples = numClassExamples
    end
  end

  local splitData = 
      torch.Tensor():typeAs(data):resize(numClasses,maxExamples,numFeatures):zero()

  for classIdx = 1, numClasses do
    local classIndicator = uniqueIDXs:eq( classIdx )
    local numClassExamples = classIndicator:sum()
    classIndicator = torch.linspace(1, numExamples, numExamples)[classIndicator]

    splitData[{classIdx,{1,numClassExamples},{}}] = 
        M.getDataFromTableOrTensor(data, classIndicator)
    --here we fill in any "extra" slots with the last value we have
    if numClassExamples < splitData:size(2) then
      for exampleIdx = numClassExamples+1, splitData:size(2) do
        splitData[{classIdx,exampleIdx}] = 
            splitData[{classIdx,numClassExamples}]
      end
    end
  end

  return splitData
end

M.getGitCommitNumAndHash = function()
	--we cache this result as soon as this script gets require'd because io.popen calls fork() which essentially copies the current process's allocated memory
	if not M.__gitCommitNumAndHash then
		local io = require 'io'
		--short version of git commit hash
		local handle = io.popen('git log -n 1 --pretty=format:"%h" ')
		local commitHash = handle:read("*a")
		handle:close()
		--tells us which commit number we are on, basically so that
		--we know the order of our commits
		local handle = io.popen('git rev-list --count HEAD')
		print(handle)
		local commitNum = handle:read("*a")
		handle:close()
		M.__gitCommitNumAndHash = commitNum:gsub("\n","") .. '_' .. commitHash:gsub("\n","")
	end
	return M.__gitCommitNumAndHash
end

M.getGitCommitNumAndHash()

M.ghettoClearStateSequential = function(model)
  --only works for nn.Container
  for m = 1, #model.modules do
    if model.modules[m].clearState then
      model.modules[m]:clearState()
    else
      model.modules[m].output = nil
      model.modules[m].gradInput = nil
      --for max pooling modules
      if model.modules[m].indices then
        model.modules[m].indices = nil
      end
    end
  end
  --finally clear out the module itself
  model.output = nil
  model.gradInput = nil
end

--this is because some networks were "delflated" 
--(using ghettoClearStateSequential) which sets fields  to nil so that they can
--be saved in a reasonable amount of space on disk.  however, once we load these
--networks, they error if the fields set to nil are not restated.
M.ghettoReinflateModel = function(model)
  for m = 1, #model.modules do
    if model.modules[m].output == nil then
      model.modules[m].output = torch.Tensor()
    end
    if model.modules[m].gradInput == nil then
      model.modules[m].gradInput = nil
    end
    --for max pooling modules
    if torch.type(model.modules[m]) == 'nn.TemporalMaxPooling' then
      if model.modules[m].indices == nil then
        model.modules[m].indices = torch.Tensor()
      end
    end
  end
  --just in case we saved a network in training mode
  model:evaluate()
end

M.fileToURI = function(file)
  --makes it so that when we print this in gnome-terminal,
  --it gets recognized as URI which we can click and open
  --from the terminal!
  return 'file://' .. file
end

M.makeConfigName = function(args, cmdOptions)

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
  elseif cmdOptions.wake_test then
	name = name .. 'WakeTest'
  else
    if cmdOptions.SO_locked then
	  name = name .. 'SOsleep'
    else 
	  name = name .. 'Sleep'
    end
  end
--per subject indicator
  if cmdOptions.run_single_subj then 
    name = name .. 'PerSubj'
  end
  if cmdOptions.float_precision then
	  name = name .. 'Single'
  end
  if cmdOptions.predict_subj then
    name = name .. 'PredSubj' .. cmdOptions.class_to_subj_loss_ratio .. 'to1'
  end
  name = name .. cmdOptions.num_hidden_mult .. 'xHidden' .. cmdOptions.num_hidden_layers 
  name = name .. '_' .. cmdOptions.ms .. 'ms'

  if args.subj_data.ERP_diff then
    name = name .. 'Diff'
  end

  --for maxPresentations, we just append "maxPres"
  if args.subj_data.max_presentations >= 1 then
    name = name .. '_maxPres' .. args.subj_data.max_presentations
  end
  
  --for "ERP_I", we have to replaced cuelocked with cuelocked_I
  if args.subj_data.ERP_I then
    name = name .. "_ERP_I"
  end

  return name
end

M.getDataFilenameFromArgs = function(args)
  local fileName = ''
  if args.subj_data.wake then
    fileNameRoot = 'wake_ERP_cuelocked_all_' .. args.subj_data.temporal_resolution .. 'ms_1000'
  elseif args.subj_data.wake_test then 
    fileNameRoot = 'waketest_all_ERP_cuelocked_all_' .. args.subj_data.temporal_resolution .. 'ms_1000'
  else
    if args.subj_data.SO_locked then
      fileNameRoot = 'sleep_ERP_SOlocked_all_phase_SO1'
    else
      fileNameRoot = 'sleep_ERP_cuelocked_all_' .. args.subj_data.temporal_resolution .. 'ms_1000'
    end
  end

  --for ERP_diff, we just replace "ERP" with "ERP_diff"
  if args.subj_data.ERP_diff then
    fileNameRoot = fileNameRoot:gsub('ERP','ERP_diff')
  end

  --for maxPresentations, we just append "maxPres"
  if args.subj_data.max_presentations >= 1 then
    fileNameRoot = fileNameRoot .. '_maxPres' .. args.subj_data.max_presentations
  end
  
  --for "ERP_I", we have to replaced cuelocked with cuelocked_I
  if args.subj_data.ERP_I then
    fileNameRoot = fileNameRoot:gsub('cuelocked','cuelocked_I')
  end

  if args.float_precision then
	  fileNameRoot = fileNameRoot .. 'Single'
  end


  if args.subj_data.sim_type > 0 then
    if args.subj_data.sim_type == 1 or args.subj_data.sim_type == 2 then
      fileName = './torch_exports/' .. fileNameRoot .. '_sim' ..  args.subj_data.sim_type .. '.mat'
    else
      error('Unknown or unimplemented simulated data type.  Only valid values are sim_type = 1 and sim_type == 2, sim_type == 3 yet to be implemented')
    end
  else
    fileName = './torch_exports/' .. fileNameRoot .. '.mat'
  end

  return fileName
end

return M
