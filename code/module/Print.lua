local Print, parent = torch.class('nn.Print', 'nn.Module')

function Print:__init(stcFlag)
    parent.__init(self)
end

function Print:updateOutput(input)
    print(input)
    self.output = input
    return self.output
end

function Print:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end
