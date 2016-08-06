--[[
    Implemented by Yannis M. Assael (www.yannisassael.com), 2016.
--]]

local GaussianNoise, parent = torch.class('nn.GaussianNoise', 'nn.Module')

function GaussianNoise:__init(std, ip)
    parent.__init(self)
    assert(type(std) == 'number', 'input is not scalar!')
    self.std = std

    -- default for inplace is false
    self.inplace = ip or false
    if (ip and type(ip) ~= 'boolean') then
        error('in-place flag must be boolean')
    end

    self.noise = torch.Tensor()
    self.train = true
end

function GaussianNoise:training()
    self.train = true
end

function GaussianNoise:evaluate()
    self.train = false
end

function GaussianNoise:updateOutput(input)
    if self.train and self.std > 0 then
        -- Generate noise
        self.noise:resizeAs(input):normal(0, self.std)

        if self.inplace then
            input:add(self.noise)
            self.output:set(input)
        else
            self.output:resizeAs(input)
            self.output:copy(input)
            self.output:add(self.noise)
        end
    else
        if self.inplace then
            self.output:set(input)
        else
            self.output:resizeAs(input)
            self.output:copy(input)
        end
    end
    return self.output
end

function GaussianNoise:updateGradInput(input, gradOutput)
    if self.inplace and self.train and self.std > 0 then
        self.gradInput:set(gradOutput)
        -- restore previous input value
        input:add(-1, self.noise)
    else
        self.gradInput:resizeAs(gradOutput)
        self.gradInput:copy(gradOutput)
    end
    return self.gradInput
end
