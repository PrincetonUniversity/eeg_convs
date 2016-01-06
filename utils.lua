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
	local job_number = os.getenv('SLURM_ARRAY_TASK_ID')
	if not job_number then
		job_number = 0
	else
		job_number = tonumber(job_number)-1
	end
	local options = sleep_eeg.param_sweep.grid({id = job_number},gridOptions)
	args.rng_seed = options.rng_seed
end

M.saveFileNameFromDriversArgs = function(args,base_name)
	local filename = ''
	local driverPrefix = base_name 
	local gitCommitHash = M.getGitCommitNumAndHash()
	local rngSeedString = 'rng_' .. args.rng_seed .. '.th7'
	filename = paths.concat(dotrc.save_dir,driverPrefix, gitCommitHash)
	if not paths.dir(filename) then
		paths.mkdir(filename)
	end
	filename = paths.concat(filename,rngSeedString)
	print(filename)
	return  filename
end


M.copyArgsRemoveUnsupportedMatioTypes = function (source)
	local tablex = require 'pl.tablex'
	local target = tablex.deepcopy(source)
	target.training.trainingIterationHooks = nil
	target.training.earlyTerminationFn = nil
	target.training.trainingCompleteHooks = nil
	return target
end





M.replaceTorchSaveWithMatSave = function(torchFilename)
	local dir = paths.dirname(torchFilename)
	local baseFilename = paths.basename(torchFilename,'.th7')
	return paths.concat(dir,baseFilename .. '.mat')
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
	return commitNum:gsub("\n","") .. '_' .. commitHash:gsub("\n","")
end

return M
