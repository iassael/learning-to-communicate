local util = {}
local log = require 'include.log'

function util.euclidean_dist(p, q)
    return math.sqrt(util.euclidean_dist2(p, q))
end

function util.euclidean_dist2(p, q)
    assert(#p == #q, 'vectors must have the same length')
    local sum = 0
    for i in ipairs(p) do
        sum = sum + (p[i] - q[i]) ^ 2
    end
    return sum
end

function util.resetParams(cur_module)
    if cur_module.modules then
        for i, module in ipairs(cur_module.modules) do
            util.resetParams(module)
        end
    else
        cur_module:reset()
    end

    return cur_module
end

function util.dc(orig)
    local orig_type = torch.type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[util.dc(orig_key)] = util.dc(orig_value)
        end
        setmetatable(copy, util.dc(getmetatable(orig)))
    elseif orig_type == 'torch.FloatTensor' or orig_type == 'torch.DoubleTensor' or orig_type == 'torch.CudaTensor' then
        -- Torch tensor
        copy = orig:clone()
    else
        -- number, string, boolean, etc
        copy = orig
    end

    return copy
end

function util.copyManyTimes(net, n)
    local nets = {}

    for i = 1, n do
        nets[#nets + 1] = util.resetParams(net:clone())
    end

    return nets
end

function util.cloneManyTimes(net, T)
    local clones = {}
    local params, gradParams = net:parameters()
    if params == nil then
        params = {}
    end
    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        local cloneParamsNoGrad
        for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
        end
        if paramsNoGrad then
            cloneParamsNoGrad = clone:parametersNoGrad()
            for i = 1, #paramsNoGrad do
                cloneParamsNoGrad[i]:set(paramsNoGrad[i])
            end
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

function util.spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys + 1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys
    if order then
        table.sort(keys, function(a, b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

function util.sprint(t)
    for k, v in util.spairs(t) do
        log.debugf('opt[\'%s\'] = %s', k, v)
    end
end

function util.f2(f)
    return string.format("%.2f", f)
end

function util.f3(f)
    return string.format("%.3f", f)
end

function util.f4(f)
    return string.format("%.4f", f)
end

function util.f5(f)
    return string.format("%.5f", f)
end

function util.f6(f)
    return string.format("%.6f", f)
end

function util.d(f)
    return string.format("%d", torch.round(f))
end



return util