--[[

    Learning to Communicate with Deep Multi-Agent Reinforcement Learning

    @article{foerster2016learning,
        title={Learning to Communicate with Deep Multi-Agent Reinforcement Learning},
        author={Foerster, Jakob N and Assael, Yannis M and de Freitas, Nando and Whiteson, Shimon},
        journal={arXiv preprint arXiv:1605.06676},
        year={2016}
    }

]] --


-- Configuration
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learning to Communicate with Deep Multi-Agent Reinforcement Learning')
cmd:text()
cmd:text('Options')

-- general options:
cmd:option('-seed', -1, 'initial random seed')
cmd:option('-threads', 1, 'number of threads')

-- gpu
cmd:option('-cuda', 0, 'cuda')

-- rl
cmd:option('-gamma', 1, 'discount factor')
cmd:option('-eps', 0.05, 'epsilon-greedy policy')

-- model
cmd:option('-model_rnn', 'gru', 'rnn type')
cmd:option('-model_dial', 0, 'use dial connection or rial')
cmd:option('-model_comm_narrow', 1, 'combines comm bits')
cmd:option('-model_know_share', 1, 'knowledge sharing')
cmd:option('-model_action_aware', 1, 'last action used as input')
cmd:option('-model_rnn_size', 128, 'rnn size')
cmd:option('-model_rnn_layers', 2, 'rnn layers')
cmd:option('-model_dropout', 0, 'dropout')
cmd:option('-model_bn', 1, 'batch normalisation')
cmd:option('-model_target', 1, 'use a target network')
cmd:option('-model_avg_q', 1, 'avearge q functions')

-- training
cmd:option('-bs', 32, 'batch size')
cmd:option('-learningrate', 5e-4, 'learningrate')
cmd:option('-nepisodes', 1e+6, 'number of episodes')
cmd:option('-nsteps', 10, 'number of steps')

cmd:option('-step', 1000, 'print every episodes')
cmd:option('-step_test', 10, 'print every episodes')
cmd:option('-step_target', 100, 'target network updates')

cmd:option('-filename', '', '')

-- games
-- ColorDigit
cmd:option('-game', 'ColorDigit', 'game name')
cmd:option('-game_dim', 28, '')
cmd:option('-game_bias', 0, '')
cmd:option('-game_colors', 2, '')
cmd:option('-game_use_mnist', 1, '')
cmd:option('-game_use_digits', 0, '')
cmd:option('-game_nagents', 2, '')
cmd:option('-game_action_space', 2, '')
cmd:option('-game_comm_limited', 0, '')
cmd:option('-game_comm_bits', 1, '')
cmd:option('-game_comm_sigma', 0, '')
cmd:option('-game_coop', 1, '')
cmd:option('-game_bottleneck', 10, '')
cmd:option('-game_level', 'extra_hard', '')
cmd:option('-game_vision_net', 'mlp', 'mlp or cnn')
cmd:option('-nsteps', 2, 'number of steps')
-- Switch
cmd:option('-game', 'Switch', 'game name')
cmd:option('-game_nagents', 3, '')
cmd:option('-game_action_space', 2, '')
cmd:option('-game_comm_limited', 1, '')
cmd:option('-game_comm_bits', 2, '')
cmd:option('-game_comm_sigma', 0, '')
cmd:option('-nsteps', 6, 'number of steps')

cmd:text()

local opt = cmd:parse(arg)

-- Custom options
if opt.seed == -1 then opt.seed = torch.random(1000000) end
opt.model_comm_narrow = opt.model_dial

if opt.model_rnn == 'lstm' then
    opt.model_rnn_states = 2 * opt.model_rnn_layers
elseif opt.model_rnn == 'gru' then
    opt.model_rnn_states = opt.model_rnn_layers
end

-- Requirements
require 'nn'
require 'optim'
local kwargs = require 'include.kwargs'
local log = require 'include.log'
local util = require 'include.util'

-- Set float as default type
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Cuda initialisation
if opt.cuda == 1 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(1)
    opt.dtype = 'torch.CudaTensor'
    print(cutorch.getDeviceProperties(1))
else
    opt.dtype = 'torch.FloatTensor'
end

if opt.model_comm_narrow == 0 and opt.game_comm_bits > 0 then
    opt.game_comm_bits = 2 ^ opt.game_comm_bits
end

-- Initialise game
local game = (require('game.' .. opt.game))(opt)

if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
    -- Without dial we add the communication actions to the action space
    opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
else
    opt.game_action_space_total = opt.game_action_space
end

-- Initialise models
local model = (require('model.' .. opt.game))(opt)

-- Print options
util.sprint(opt)

-- Model target evaluate
model.evaluate(model.agent_target)

-- Get parameters
local params, gradParams, params_target, _ = model.getParameters()

-- Optimisation function
local optim_func, optim_config = model.optim()
local optim_state = {}

-- Initialise agents
local agent = {}
for i = 1, opt.game_nagents do
    agent[i] = {}

    agent[i].id = torch.Tensor():type(opt.dtype):resize(opt.bs):fill(i)

    -- Populate init state
    agent[i].input = {}
    agent[i].input_target = {}
    agent[i].state = {}
    agent[i].state_target = {}
    agent[i].state[0] = {}
    agent[i].state_target[0] = {}
    for j = 1, opt.model_rnn_states do
        agent[i].state[0][j] = torch.zeros(opt.bs, opt.model_rnn_size):type(opt.dtype)
        agent[i].state_target[0][j] = torch.zeros(opt.bs, opt.model_rnn_size):type(opt.dtype)
    end

    agent[i].d_state = {}
    agent[i].d_state[0] = {}
    for j = 1, opt.model_rnn_states do
        agent[i].d_state[0][j] = torch.zeros(opt.bs, opt.model_rnn_size):type(opt.dtype)
    end

    -- Store q values
    agent[i].q_next_max = {}
    agent[i].q_comm_next_max = {}
end

local episode = {}

-- Initialise aux vectors
local d_err = torch.Tensor(opt.bs, opt.game_action_space_total):type(opt.dtype)
local td_err = torch.Tensor(opt.bs):type(opt.dtype)
local td_comm_err = torch.Tensor(opt.bs):type(opt.dtype)
local stats = {
    r_episode = torch.zeros(opt.nsteps),
    td_err = torch.zeros(opt.step),
    td_comm = torch.zeros(opt.step),
    train_r = torch.zeros(opt.step, opt.game_nagents),
    steps = torch.zeros(opt.step / opt.step_test),
    test_r = torch.zeros(opt.step / opt.step_test, opt.game_nagents),
    comm_per = torch.zeros(opt.step / opt.step_test),
    te = torch.zeros(opt.step)
}

local replay = {}

-- Run episode
local function run_episode(opt, game, model, agent, test_mode)

    -- Test mode
    test_mode = test_mode or false

    -- Reset game
    game:reset()

    -- Initialise episode
    local step = 1
    local episode = {
        comm_per = torch.zeros(opt.bs),
        r = torch.zeros(opt.bs, opt.game_nagents),
        steps = torch.zeros(opt.bs),
        ended = torch.zeros(opt.bs),
        comm_count = 0,
        non_comm_count = 0
    }
    episode[step] = {
        s_t = game:getState(),
        terminal = torch.zeros(opt.bs)
    }
    if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
        episode[step].comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
        if opt.model_dial == 1 and opt.model_target == 1 then
            episode[step].comm_target = episode[step].comm:clone()
        end
        episode[step].d_comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
    end


    -- Run for N steps
    local steps = test_mode and opt.nsteps or opt.nsteps + 1
    while step <= steps and episode.ended:sum() < opt.bs do

        -- Initialise next step
        episode[step + 1] = {}

        -- Initialise comm channel
        if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
            episode[step + 1].comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
            episode[step + 1].d_comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
            if opt.model_dial == 1 and opt.model_target == 1 then
                episode[step + 1].comm_target = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
            end
        end

        -- Forward pass
        episode[step].a_t = torch.zeros(opt.bs, opt.game_nagents):type(opt.dtype)
        if opt.model_dial == 0 then
            episode[step].a_comm_t = torch.zeros(opt.bs, opt.game_nagents):type(opt.dtype)
        end

        -- Iterate agents
        for i = 1, opt.game_nagents do
            agent[i].input[step] = {
                episode[step].s_t[i],
                agent[i].id,
                agent[i].state[step - 1]
            }

            -- Communication enabled
            if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
                local comm_limited = game:getCommLimited(step, i)
                local comm = episode[step].comm:clone()
                if comm_limited then
                    -- Create limited communication channel nbits
                    local comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits):type(opt.dtype)
                    for b = 1, opt.bs do
                        if comm_limited[b] == 0 then
                            comm_lim[{ { b } }]:zero()
                        else
                            comm_lim[{ { b } }] = comm[{ { b }, unpack(comm_limited[b]) }]
                        end
                    end
                    table.insert(agent[i].input[step], comm_lim)
                else
                    -- zero out own communication if not action aware
                    comm[{ {}, { i } }]:zero()
                    table.insert(agent[i].input[step], comm)
                end
            end

            -- Last action enabled
            if opt.model_action_aware == 1 then
                -- If comm always then use both action
                if opt.model_dial == 0 then
                    local la = { torch.ones(opt.bs), torch.ones(opt.bs) }
                    if step > 1 then
                        for b = 1, opt.bs do
                            -- Last action
                            if episode[step - 1].a_t[b][i] > 0 then
                                la[1][{ { b } }] = episode[step - 1].a_t[b][i] + 1
                            end
                            -- Last comm action
                            if episode[step - 1].a_comm_t[b][i] > 0 then
                                la[2][{ { b } }] = episode[step - 1].a_comm_t[b][i] - opt.game_action_space + 1
                            end
                        end
                    end
                    table.insert(agent[i].input[step], la)
                else
                    -- Action aware for single a, comm action
                    local la = torch.ones(opt.bs)
                    if step > 1 then
                        for b = 1, opt.bs do
                            if episode[step - 1].a_t[b][i] > 0 then
                                la[{ { b } }] = episode[step - 1].a_t[b][i] + 1
                            end
                        end
                    end
                    table.insert(agent[i].input[step], la)
                end
            end

            -- Compute Q values
            local comm, state, q_t
            agent[i].state[step], q_t = unpack(model.agent[model.id(step, i)]:forward(agent[i].input[step]))


            -- If dial split out the comm values from q values
            if opt.model_dial == 1 then
                q_t, comm = DRU(q_t, test_mode)
            end

            -- Pick an action (epsilon-greedy)
            local action_range, action_range_comm
            local max_value, max_a, max_a_comm
            if opt.model_dial == 0 then
                action_range, action_range_comm = game:getActionRange(step, i)
            else
                action_range = game:getActionRange(step, i)
            end

            -- If Limited action range
            if action_range then
                agent[i].range = agent[i].range or torch.range(1, opt.game_action_space_total)
                max_value = torch.Tensor(opt.bs, 1)
                max_a = torch.zeros(opt.bs, 1)
                if opt.model_dial == 0 then
                    max_a_comm = torch.zeros(opt.bs, 1)
                end
                for b = 1, opt.bs do
                    -- If comm always fetch range for comm and actions
                    if opt.model_dial == 0 then
                        -- If action was taken
                        if action_range[b][2][1] > 0 then
                            local v, a = torch.max(q_t[action_range[b]], 2)
                            max_value[b] = v:squeeze()
                            max_a[b] = agent[i].range[{ action_range[b][2] }][a:squeeze()]
                        end
                        -- If comm action was taken
                        if action_range_comm[b][2][1] > 0 then
                            local v, a = torch.max(q_t[action_range_comm[b]], 2)
                            max_a_comm[b] = agent[i].range[{ action_range_comm[b][2] }][a:squeeze()]
                        end
                    else
                        local v, a = torch.max(q_t[action_range[b]], 2)
                        max_a[b] = agent[i].range[{ action_range[b][2] }][a:squeeze()]
                    end
                end
            else
                -- If comm always pick max_a and max_comm
                if opt.model_dial == 0 and opt.game_comm_bits > 0 then
                    _, max_a = torch.max(q_t[{ {}, { 1, opt.game_action_space } }], 2)
                    _, max_a_comm = torch.max(q_t[{ {}, { opt.game_action_space + 1, opt.game_action_space_total } }], 2)
                    max_a_comm = max_a_comm + opt.game_action_space
                else
                    _, max_a = torch.max(q_t, 2)
                end
            end

            -- Store actions
            episode[step].a_t[{ {}, { i } }] = max_a
            if opt.model_dial == 0 and opt.game_comm_bits > 0 then
                episode[step].a_comm_t[{ {}, { i } }] = max_a_comm
            end

            for b = 1, opt.bs do

                -- Epsilon-greedy action picking
                if not test_mode then
                    if opt.model_dial == 0 then
                        -- Random action
                        if torch.uniform() < opt.eps then
                            if action_range then
                                if action_range[b][2][1] > 0 then
                                    local a_range = agent[i].range[{ action_range[b][2] }]
                                    local a_idx = torch.random(a_range:nElement())
                                    episode[step].a_t[b][i] = agent[i].range[{ action_range[b][2] }][a_idx]
                                end
                            else
                                episode[step].a_t[b][i] = torch.random(opt.game_action_space)
                            end
                        end

                        -- Random communication
                        if opt.game_comm_bits > 0 and torch.uniform() < opt.eps then
                            if action_range then
                                if action_range_comm[b][2][1] > 0 then
                                    local a_range = agent[i].range[{ action_range_comm[b][2] }]
                                    local a_idx = torch.random(a_range:nElement())
                                    episode[step].a_comm_t[b][i] = agent[i].range[{ action_range_comm[b][2] }][a_idx]
                                end
                            else
                                episode[step].a_comm_t[b][i] = torch.random(opt.game_action_space + 1, opt.game_action_space_total)
                            end
                        end

                    else
                        if torch.uniform() < opt.eps then
                            if action_range then
                                local a_range = agent[i].range[{ action_range[b][2] }]
                                local a_idx = torch.random(a_range:nElement())
                                episode[step].a_t[b][i] = agent[i].range[{ action_range[b][2] }][a_idx]
                            else
                                episode[step].a_t[b][i] = torch.random(q_t[b]:size(1))
                            end
                        end
                    end
                end

                -- If communication action populate channel
                if step <= opt.nsteps then
                    -- For dial we 'forward' the direct activation otherwise we shift the a_t into the 1-game_comm_bits range
                    if opt.model_dial == 1 then
                        episode[step + 1].comm[b][i] = comm[b]
                    else
                        local a_t = episode[step].a_comm_t[b][i] - opt.game_action_space
                        if a_t > 0 then
                            episode[step + 1].comm[b][{ { i }, { a_t } }] = 1
                        end
                    end

                    if episode.ended[b] == 0 then
                        episode.comm_per[{ { b } }]:add(1 / opt.game_nagents)
                    end
                    episode.comm_count = episode.comm_count + 1

                else
                    episode.non_comm_count = episode.non_comm_count + 1
                end
            end
        end

        -- Compute reward for current state-action pair
        episode[step].r_t, episode[step].terminal = game:step(episode[step].a_t)


        -- Accumulate steps (not for +1 step)
        if step <= opt.nsteps then
            for b = 1, opt.bs do
                if episode.ended[b] == 0 then

                    -- Keep steps and rewards
                    episode.steps[{ { b } }]:add(1)
                    episode.r[{ { b } }]:add(episode[step].r_t[b])

                    -- Check if terminal
                    if episode[step].terminal[b] == 1 then
                        episode.ended[{ { b } }] = 1
                    end
                end
            end
        end


        -- Target Network, for look-ahead
        if opt.model_target == 1 and not test_mode then
            for i = 1, opt.game_nagents do
                local comm = agent[i].input[step][4]

                if opt.game_comm_bits > 0 and opt.game_nagents > 1 and opt.model_dial == 1 then
                    local comm_limited = game:getCommLimited(step, i)
                    comm = episode[step].comm_target:clone()

                    -- Create limited communication channel nbits
                    if comm_limited then
                        local comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
                        for b = 1, opt.bs do
                            if comm_limited[b] == 0 then
                                comm_lim[{ { b } }] = 0
                            else
                                comm_lim[{ { b } }] = comm[{ { b }, unpack(comm_limited[b]) }]
                            end
                        end
                        comm = comm_lim
                    else
                        comm[{ {}, { i } }] = 0
                    end
                end

                -- Target input
                agent[i].input_target[step] = {
                    agent[i].input[step][1],
                    agent[i].input[step][2],
                    agent[i].state_target[step - 1],
                    comm,
                    agent[i].input[step][5],
                }

                -- Forward target
                local state, q_t_target = unpack(model.agent_target[model.id(step, i)]:forward(agent[i].input_target[step]))
                agent[i].state_target[step] = state
                if opt.model_dial == 1 then
                    q_t_target, comm = DRU(q_t_target, test_mode)
                end

                -- Limit actions
                if opt.model_dial == 0 and opt.game_comm_bits > 0 then
                    local action_range, action_range_comm = game:getActionRange(step, i)
                    if action_range then
                        agent[i].q_next_max[step] = torch.zeros(opt.bs)
                        agent[i].q_comm_next_max[step] = torch.zeros(opt.bs)
                        for b = 1, opt.bs do
                            if action_range[b][2][1] > 0 then
                                agent[i].q_next_max[step][b], _ = torch.max(q_t_target[action_range[b]], 2)
                            else
                                error('Not implemented')
                            end

                            -- If comm not available pick from None
                            if action_range_comm[b][2][1] > 0 then
                                agent[i].q_comm_next_max[step][b], _ = torch.max(q_t_target[action_range_comm[b]], 2)
                            else
                                agent[i].q_comm_next_max[step][b], _ = torch.max(q_t_target[action_range[b]], 2)
                            end
                        end
                    else
                        agent[i].q_next_max[step], _ = torch.max(q_t_target[{ {}, { 1, opt.game_action_space } }], 2)
                        agent[i].q_comm_next_max[step], _ = torch.max(q_t_target[{ {}, { opt.game_action_space + 1, opt.game_action_space_total } }], 2)
                    end
                else
                    local action_range = game:getActionRange(step, i)
                    if action_range then
                        agent[i].q_next_max[step] = torch.zeros(opt.bs)
                        for b = 1, opt.bs do
                            if action_range[b][2][1] > 0 then
                                agent[i].q_next_max[step][b], _ = torch.max(q_t_target[action_range[b]], 2)
                            end
                        end
                    else
                        agent[i].q_next_max[step], _ = torch.max(q_t_target, 2)
                    end
                end

                if opt.model_dial == 1 then
                    for b = 1, opt.bs do
                        episode[step + 1].comm_target[b][i] = comm[b]
                    end
                end
            end
        end

        -- Forward next step
        step = step + 1
        if episode.ended:sum() < opt.bs then
            episode[step].s_t = game:getState()
        end
    end

    -- Update stats
    episode.nsteps = episode.steps:max()
    episode.comm_per:cdiv(episode.steps)

    return episode, agent
end


-- split out the communication bits and add noise.
function DRU(q_t, test_mode)
    if opt.model_dial == 0 then error('Warning!! Should only be used in DIAL') end
    local bound = opt.game_action_space

    local q_t_n = q_t[{ {}, { 1, bound } }]:clone()
    local comm = q_t[{ {}, { bound + 1, opt.game_action_space_total } }]:clone()
    if test_mode then
        if opt.model_comm_narrow == 0 then
            local ind
            _, ind = torch.max(comm, 2)
            comm:zero()
            for b = 1, opt.bs do
                comm[b][ind[b][1]] = 20
            end
        else
            comm = comm:gt(0.5):type(opt.dtype):add(-0.5):mul(2 * 20)
        end
    end
    if opt.game_comm_sigma > 0 and not test_mode then
        local noise_vect = torch.randn(comm:size()):type(opt.dtype):mul(opt.game_comm_sigma)
        comm = comm + noise_vect
    end
    return q_t_n, comm
end

-- Start time
local beginning_time = torch.tic()

-- Iterate episodes
for e = 1, opt.nepisodes do

    stats.e = e

    -- Initialise clock
    local time = sys.clock()

    -- Model training
    model.training(model.agent)

    -- Run episode
    episode, agent = run_episode(opt, game, model, agent)

    -- Rewards stats
    stats.train_r[(e - 1) % opt.step + 1] = episode.r:mean(1)

    -- Reset parameters
    if e == 1 then
        gradParams:zero()
    end

    -- Backwawrd pass
    local step_back = 1
    for step = episode.nsteps, 1, -1 do
        stats.td_err[(e - 1) % opt.step + 1] = 0
        stats.td_comm[(e - 1) % opt.step + 1] = 0

        -- Iterate agents
        for i = 1, opt.game_nagents do

            -- Compute Q values
            local state, q_t = unpack(model.agent[model.id(step, i)].output)

            -- Compute td error
            td_err:zero()
            td_comm_err:zero()
            d_err:zero()

            for b = 1, opt.bs do
                if step >= episode.steps[b] then
                    -- if first backward init RNN
                    for j = 1, opt.model_rnn_states do
                        agent[i].d_state[step_back - 1][j][b]:zero()
                    end
                end

                if step <= episode.steps[b] then

                    -- if terminal state or end state => no future rewards
                    if episode[step].a_t[b][i] > 0 then
                        if episode[step].terminal[b] == 1 then
                            td_err[b] = episode[step].r_t[b][i] - q_t[b][episode[step].a_t[b][i]]
                        else
                            local q_next_max
                            if opt.model_avg_q == 1 and opt.model_dial == 0 and episode[step].a_comm_t[b][i] > 0 then
                                q_next_max = (agent[i].q_next_max[step + 1]:squeeze() + agent[i].q_comm_next_max[step + 1]:squeeze()) / 2
                            else
                                q_next_max = agent[i].q_next_max[step + 1]:squeeze()
                            end
                            td_err[b] = episode[step].r_t[b][i] + opt.gamma * q_next_max[b] - q_t[b][episode[step].a_t[b][i]]
                        end
                        d_err[{ { b }, { episode[step].a_t[b][i] } }] = -td_err[b]

                    else
                        error('Error!')
                    end

                    -- Delta Q for communication
                    if opt.model_dial == 0 then
                        if episode[step].a_comm_t[b][i] > 0 then
                            if episode[step].terminal[b] == 1 then
                                td_comm_err[b] = episode[step].r_t[b][i] - q_t[b][episode[step].a_comm_t[b][i]]
                            else
                                local q_next_max
                                if opt.model_avg_q == 1 and episode[step].a_t[b][i] > 0 then
                                    q_next_max = (agent[i].q_next_max[step + 1]:squeeze() + agent[i].q_comm_next_max[step + 1]:squeeze()) / 2
                                else
                                    q_next_max = agent[i].q_comm_next_max[step + 1]:squeeze()
                                end
                                td_comm_err[b] = episode[step].r_t[b][i] + opt.gamma * q_next_max[b] - q_t[b][episode[step].a_comm_t[b][i]]
                            end
                            d_err[{ { b }, { episode[step].a_comm_t[b][i] } }] = -td_comm_err[b]
                        end
                    end

                    -- If we use dial and the agent took the umbrella comm action and the messsage happened before last round, the we get incoming derivaties
                    if opt.model_dial == 1 and step < episode.steps[b] then
                        -- Derivatives with respect to agent_i's message are stored in d_comm[b][i]
                        local bound = opt.game_action_space
                        d_err[{ { b }, { bound + 1, opt.game_action_space_total } }]:add(episode[step + 1].d_comm[b][i])
                    end
                end
            end

            -- Track td-err
            stats.td_err[(e - 1) % opt.step + 1] = stats.td_err[(e - 1) % opt.step + 1] + 0.5 * td_err:clone():pow(2):mean()
            if opt.model_dial == 0 then
                stats.td_comm[(e - 1) % opt.step + 1] = stats.td_comm[(e - 1) % opt.step + 1] + 0.5 * td_comm_err:clone():pow(2):mean()
            end

            -- Track the amplitude of the dial-derivatives
            if opt.model_dial == 1 then
                local bound = opt.game_action_space
                stats.td_comm[(e - 1) % opt.step + 1] = stats.td_comm[(e - 1) % opt.step + 1] + 0.5 * d_err[{ {}, { bound + 1, opt.game_action_space_total } }]:clone():pow(2):mean()
            end

            -- Backward pass
            local grad = model.agent[model.id(step, i)]:backward(agent[i].input[step], {
                agent[i].d_state[step_back - 1],
                d_err
            })

            --'state' is the 3rd input, so we can extract d_state
            agent[i].d_state[step_back] = grad[3]

            --For dial we need to write add the derivatives w/ respect to the incoming messages to the d_comm tracker
            if opt.model_dial == 1 then
                local comm_limited = game:getCommLimited(step, i)
                local comm_grad = grad[4]

                if comm_limited then
                    for b = 1, opt.bs do
                        -- Agent could only receive the message if they were active
                        if comm_limited[b] ~= 0 then
                            episode[step].d_comm[{ { b }, unpack(comm_limited[b]) }]:add(comm_grad[b])
                        end
                    end
                else
                    -- zero out own communication unless it's part of the switch riddle
                    comm_grad[{ {}, { i } }]:zero()
                    episode[step].d_comm:add(comm_grad)
                end
            end
        end

        -- Count backward steps
        step_back = step_back + 1
    end

    -- Update gradients
    local feval = function(x)

        -- Normalise Gradients
        gradParams:div(opt.game_nagents * opt.bs)

        -- Clip Gradients
        gradParams:clamp(-10, 10)

        return nil, gradParams
    end

    optim_func(feval, params, optim_config, optim_state)

    -- Gradient statistics
    if e % opt.step == 0 then
        stats.grad_norm = gradParams:norm() / gradParams:nElement() * 1000
    end

    -- Reset parameters
    gradParams:zero()

    -- Update target network
    if e % opt.step_target == 0 then
        params_target:copy(params)
    end

    -- Test
    if e % opt.step_test == 0 then
        local test_idx = (e / opt.step_test - 1) % (opt.step / opt.step_test) + 1

        local episode, _ = run_episode(opt, game, model, agent, true)
        stats.test_r[test_idx] = episode.r:mean(1)
        stats.steps[test_idx] = episode.steps:mean()
        stats.comm_per[test_idx] = episode.comm_count / (episode.comm_count + episode.non_comm_count)
    end

    -- Compute statistics
    stats.te[(e - 1) % opt.step + 1] = sys.clock() - time

    if e == opt.step then
        stats.td_err_avg = stats.td_err:mean()
        stats.td_comm_avg = stats.td_comm:mean()
        stats.train_r_avg = stats.train_r:mean(1)
        stats.test_r_avg = stats.test_r:mean(1)
        stats.steps_avg = stats.steps:mean()
        stats.comm_per_avg = stats.comm_per:mean()
        stats.te_avg = stats.te:mean()
    elseif e % opt.step == 0 then
        local coef = 0.9
        stats.td_err_avg = stats.td_err_avg * coef + stats.td_err:mean() * (1 - coef)
        stats.td_comm_avg = stats.td_comm_avg * coef + stats.td_comm:mean() * (1 - coef)
        stats.train_r_avg = stats.train_r_avg * coef + stats.train_r:mean(1) * (1 - coef)
        stats.test_r_avg = stats.test_r_avg * coef + stats.test_r:mean(1) * (1 - coef)
        stats.steps_avg = stats.steps_avg * coef + stats.steps:mean() * (1 - coef)
        stats.comm_per_avg = stats.comm_per_avg * coef + stats.comm_per:mean() * (1 - coef)
        stats.te_avg = stats.te_avg * coef + stats.te:mean() * (1 - coef)
    end

    -- Print statistics
    if e % opt.step == 0 then
        log.infof('e=%d, td_err=%.3f, td_err_avg=%.3f, td_comm=%.3f, td_comm_avg=%.3f, tr_r=%.2f, tr_r_avg=%.2f, te_r=%.2f, te_r_avg=%.2f, st=%.1f, comm=%.1f%%, grad=%.3f, t/s=%.2f s, t=%d m',
            stats.e,
            stats.td_err:mean(),
            stats.td_err_avg,
            stats.td_comm:mean(),
            stats.td_comm_avg,
            stats.train_r:mean(),
            stats.train_r_avg:mean(),
            stats.test_r:mean(),
            stats.test_r_avg:mean(),
            stats.steps_avg,
            stats.comm_per_avg * 100,
            stats.grad_norm,
            stats.te_avg * opt.step,
            torch.toc(beginning_time) / 60)

        collectgarbage()
    end

    -- run model specific statistics
    model.stats(opt, game, stats, e)

    -- run model specific statistics
    model.save(opt, stats, model)
end
