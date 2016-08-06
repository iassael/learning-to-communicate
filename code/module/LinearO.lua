--[[
    Torch Linear Unit with Orthogonal Weight Initialization

    Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    http://arxiv.org/abs/1312.6120
    
    Implemented by Yannis M. Assael (www.yannisassael.com), 2016.

]] --

local LinearO, parent = torch.class('nn.LinearO', 'nn.Linear')

function LinearO:__init(inputSize, outputSize)
    parent.__init(self, inputSize, outputSize)
    self:reset()
end

function LinearO:reset()
    local initScale = 1.1 -- math.sqrt(2)

    local M1 = torch.randn(self.weight:size(1), self.weight:size(1))
    local M2 = torch.randn(self.weight:size(2), self.weight:size(2))

    local n_min = math.min(self.weight:size(1), self.weight:size(2))

    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)

    self.weight:copy(Q1:narrow(2, 1, n_min) * Q2:narrow(1, 1, n_min)):mul(initScale)

    self.bias:zero()
end

