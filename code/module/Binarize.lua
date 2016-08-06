local Binarize, parent = torch.class('nn.Binarize', 'nn.Module')

function Binarize:__init(stcFlag)
    parent.__init(self)
    self.stcFlag = stcFlag
    self.randmat = torch.Tensor();
    self.outputR = torch.Tensor();
end

function Binarize:updateOutput(input)
    self.randmat:resizeAs(input);
    self.outputR:resizeAs(input);
    self.output:resizeAs(input);
    self.outputR:copy(input):add(1):div(2)
    if self.train and self.stcFlag then
        local mask = self.outputR - self.randmat:rand(self.randmat:size())
        self.output = mask:sign()
    else
        self.output:copy(self.outputR):add(-0.5):sign()
    end
    return self.output
end

function Binarize:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput) --:mul(0.5)
    return self.gradInput
end
