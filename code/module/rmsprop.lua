-- RMSProp with momentum as found in "Generating Sequences With Recurrent Neural Networks"
function optim.rmspropm(opfunc, x, config, state)
    -- Get state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local momentum = config.momentum or 0.95
    local epsilon = config.epsilon or 0.01

    -- Evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- Initialise storage
    if not state.g then
        state.g = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.gSq = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
    end

    -- g = αg + (1 - α)df/dx
    state.g:mul(momentum):add(1 - momentum, dfdx) -- Calculate momentum
    -- tmp = df/dx . df/dx
    state.tmp:cmul(dfdx, dfdx)
    -- gSq = αgSq + (1 - α)df/dx
    state.gSq:mul(momentum):add(1 - momentum, state.tmp) -- Calculate "squared" momentum
    -- tmp = g . g
    state.tmp:cmul(state.g, state.g)
    -- tmp = (-tmp + gSq + ε)^0.5
    state.tmp:neg():add(state.gSq):add(epsilon):sqrt()

    -- Update x = x - lr x df/dx / tmp
    x:addcdiv(-lr, dfdx, state.tmp)

    -- Return x*, f(x) before optimisation
    return x, { fx }
end
