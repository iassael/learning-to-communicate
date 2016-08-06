require 'torch'
local class = require 'class'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'


local ColorDigit = class('ColorDigit')

function ColorDigit:__init(opt)
    self.opt = opt

    self.step_counter = 1
    -- Preprocess data
    local dataset = (require 'mnist').traindataset()
    local data = {}
    local lookup = {}
    for i = 1, dataset.size do
        -- Shift 0 class
        local y = dataset[i].y + 1
        -- Create array
        if not data[y] then
            data[y] = {}
        end
        -- Move data
        data[y][#data[y] + 1] = {
            x = dataset[i].x,
            y = y
        }
        lookup[i] = y
    end

    self.mnist = data

    self.mnist.lookup = lookup

    -- Rewards
    self.reward = torch.zeros(self.opt.bs)
    self.terminal = torch.zeros(self.opt.bs)

    -- Spawn new game
    self:reset()
end

function ColorDigit:loadDigit()
    -- Pick random digit and color
    local color_id = torch.zeros(self.opt.bs)
    local number = torch.zeros(self.opt.bs)
    local x = torch.zeros(self.opt.bs, self.opt.game_colors, self.opt.game_dim, self.opt.game_dim):type(self.opt.dtype)
    for b = 1, self.opt.bs do
        -- Pick number
        local num
        if self.opt.game_use_mnist == 1 then
            local index = torch.random(#self.mnist.lookup)
            num = self.mnist.lookup[index]
        elseif torch.uniform() < self.opt.game_bias then
            num = 1
        else
            num = torch.random(10)
        end

        number[b] = num

        -- Pick color
        color_id[b] = torch.random(self.opt.game_colors)

        -- Pick dataset id
        local id = torch.random(#self.mnist[num])
        x[b][color_id[b]] = self.mnist[num][id].x
    end
    return { x, color_id, number }
end


function ColorDigit:reset()

    -- Load images
    self.state = { self:loadDigit(), self:loadDigit() }

    -- Reset rewards
    self.reward:zero()
    self.terminal:zero()

    -- Reset counter
    self.step_counter = 1

    return self
end

function ColorDigit:getActionRange()
    return nil
end

function ColorDigit:getCommLimited()
    return nil
end

function ColorDigit:getReward(a)

    local color_1 = self.state[1][2]
    local color_2 = self.state[2][2]
    local digit_1 = self.state[1][3]
    local digit_2 = self.state[2][3]

    local reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    for b = 1, self.opt.bs do
        if self.opt.game_level == "extra_hard_local" then
            if a[b][2] <= self.opt.game_action_space and self.step_counter > 1 then
                reward[b] = 2 * (-1) ^ (digit_1[b] + a[b][2] + color_2[b]) + (-1) ^ (digit_2[b] + a[b][2] + color_1[b])
            end
            if a[b][1] <= self.opt.game_action_space and self.step_counter > 1 then
                reward[b] = reward[b] + 2 * (-1) ^ (digit_2[b] + a[b][1] + color_1[b]) + (-1) ^ (digit_1[b] + a[b][1] + color_2[b])
            end
        elseif self.opt.game_level == "many_bits" then
            if a[b][1] <= self.opt.game_action_space and self.step_counter == self.opt.nsteps then
                if digit_2[b] == a[b][1] then
                    reward[b] = reward[b] + 0.5
                end
            end

            if a[b][2] <= self.opt.game_action_space and self.step_counter == self.opt.nsteps then
                if digit_1[b] == a[b][2] then
                    reward[b] = reward[b] + 0.5
                end
            end
        else
            error("[ColorDigit] wrong level")
        end
    end

    local reward_coop = torch.zeros(self.opt.bs, self.opt.game_nagents)
    reward_coop[{ {}, { 2 } }] = (reward[{ {}, { 2 } }] + reward[{ {}, { 1 } }] * self.opt.game_coop) / (1 + self.opt.game_coop)
    reward_coop[{ {}, { 1 } }] = (reward[{ {}, { 1 } }] + reward[{ {}, { 2 } }] * self.opt.game_coop) / (1 + self.opt.game_coop)

    return reward_coop
end

function ColorDigit:step(a)
    local reward, terminal

    reward = self:getReward(a)

    if self.step_counter == self.opt.nsteps then
        self.terminal:fill(1)
    end

    self.step_counter = self.step_counter + 1

    return reward, self.terminal:clone()
end


function ColorDigit:getState()
    if self.opt.game_use_digits == 1 then
        return { self.state[1][3], self.state[2][3] }
    else
        return { self.state[1][1], self.state[2][1] }
    end
end

return ColorDigit

