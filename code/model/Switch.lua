require 'nn'
require 'nngraph'
require 'optim'
require 'csvigo'

local kwargs = require 'include.kwargs'
local log = require 'include.log'
local util = require 'include.util'
local LSTM = require 'module.LSTM'
local GRU = require 'module.GRU'
require 'module.rmsprop'
require 'module.GaussianNoise'
require 'module.Binarize'

return function(opt)

    local exp = {}

    function exp.optim(iter)
        -- iter can be used for learning rate decay
        -- local optimfunc = optim.adam
        local optimfunc = optim.rmsprop
        local optimconfig = { learningRate = opt.learningrate }
        return optimfunc, optimconfig
    end

    function exp.save(opt, stats, model)
        if stats.e % opt.step == 0 then
            if opt.filename == '' then
                exp.save_path = exp.save_path or paths.concat('results', opt.game .. '_' .. opt.game_nagents ..
                        (opt.model_dial == 1 and '_dial' or '') .. '_' .. string.upper(string.format("%x", opt.seed)))
            else
                exp.save_path = exp.save_path or paths.concat('results', opt.game .. '_' .. opt.game_nagents ..
                        (opt.model_dial == 1 and '_dial' or '') .. '_' .. opt.filename .. '_' .. string.upper(string.format("%x", opt.seed)))
            end


            -- Save opt
            if stats.e == opt.step then
                os.execute('mkdir -p ' .. exp.save_path)
                local opt_csv = {}
                for k, v in util.spairs(opt) do
                    table.insert(opt_csv, { k, v })
                end

                csvigo.save({
                    path = paths.concat(exp.save_path, 'opt.csv'),
                    data = opt_csv,
                    verbose = false
                })
            end

            -- keep stats
            stats.history = stats.history or { { 'e', 'td_err', 'td_comm', 'train_r', 'test_r', 'test_opt', 'test_god', 'steps', 'comm_per', 'te' } }
            table.insert(stats.history, {
                stats.e,
                stats.td_err:mean(),
                stats.td_comm:mean(),
                stats.train_r:mean(),
                stats.test_r:mean(),
                stats.test_opt:mean(),
                stats.test_god:mean(),
                stats.steps:mean(),
                stats.comm_per:mean(),
                stats.te:mean()
            })

            -- Save stats csv
            csvigo.save({
                path = paths.concat(exp.save_path, 'stats.csv'),
                data = stats.history,
                verbose = false
            })

            -- Save action histogram
            if opt.hist_action == 1 then
                -- Append to memory
                stats.history_hist_action = stats.history_hist_action or {}
                table.insert(stats.history_hist_action,
                    stats.hist_action_avg:totable()[1])

                -- save csv
                csvigo.save({
                    path = paths.concat(exp.save_path, 'hist_action.csv'),
                    data = stats.history_hist_action,
                    verbose = false
                })
            end

            -- Save action histogram
            if opt.hist_comm == 1 then
                -- Append to memory
                stats.history_hist_comm = stats.history_hist_comm or {}
                table.insert(stats.history_hist_comm,
                    stats.hist_comm_avg:totable()[1])

                -- save csv
                csvigo.save({
                    path = paths.concat(exp.save_path, 'hist_comm.csv'),
                    data = stats.history_hist_comm,
                    verbose = false
                })
            end
            -- save model
            if stats.e % (opt.step * 10) == 0 then
                log.debug('Saving model')

                -- clear state
                -- exp.clearState(model.agent)

                -- save model
                local filename = paths.concat(exp.save_path, 'exp.t7')
                torch.save(filename, { opt, stats, model.agent })
            end
        end
    end

    function exp.load()
    end

    function exp.clearState(model)
        for i = 1, #model do
            model[i]:clearState()
        end
    end

    function exp.training(model)
        for i = 1, #model do
            model[i]:training()
        end
    end

    function exp.evaluate(model)
        for i = 1, #model do
            model[i]:evaluate()
        end
    end

    function exp.getParameters()
        -- Get model params
        local a = nn.Container()
        for i = 1, #exp.agent do
            a:add(exp.agent[i])
        end
        local params, gradParams = a:getParameters()

        log.infof('Creating model(s), params=%d', params:nElement())

        -- Get target model params
        local a = nn.Container()
        for i = 1, #exp.agent_target do
            a:add(exp.agent_target[i])
        end
        local params_target, gradParams_target = a:getParameters()

        log.infof('Creating target model(s), params=%d', params_target:nElement())

        return params, gradParams, params_target, gradParams_target
    end

    function exp.id(step_i, agent_i)
        return (step_i - 1) * opt.game_nagents + agent_i
    end

    function exp.stats(opt, game, stats, e)

        if e % opt.step_test == 0 then
            local test_idx = (e / opt.step_test - 1) % (opt.step / opt.step_test) + 1

            -- Initialise
            stats.test_opt = stats.test_opt or torch.zeros(opt.step / opt.step_test, opt.game_nagents)
            stats.test_god = stats.test_god or torch.zeros(opt.step / opt.step_test, opt.game_nagents)

            -- Naive strategy
            local r_naive = 0
            for b = 1, opt.bs do
                local has_been = game.has_been[{ { b }, { 1, opt.nsteps }, {} }]:sum(2):squeeze(2):gt(0):float():sum()
                if has_been == opt.game_nagents then
                    r_naive = r_naive + game.reward_all_live
                else
                    r_naive = r_naive + game.reward_all_die
                end
            end
            stats.test_opt[test_idx] = r_naive / opt.bs

            -- God strategy
            local r_god = 0
            for b = 1, opt.bs do
                local has_been = game.has_been[{ { b }, { 1, opt.nsteps }, {} }]:sum(2):squeeze(2):gt(0):float():sum()
                if has_been == opt.game_nagents then
                    r_god = r_god + game.reward_all_live
                end
            end
            stats.test_god[test_idx] = r_god / opt.bs
        end

        -- Keep stats
        if e == opt.step then
            stats.test_opt_avg = stats.test_opt:mean()
            stats.test_god_avg = stats.test_god:mean()
        elseif e % opt.step == 0 then
            local coef = 0.9
            stats.test_opt_avg = stats.test_opt_avg * coef + stats.test_opt:mean() * (1 - coef)
            stats.test_god_avg = stats.test_god_avg * coef + stats.test_god:mean() * (1 - coef)
        end

        -- Print statistics
        if e % opt.step == 0 then
            log.infof('te_opt=%.2f, te_opt_avg=%.2f, te_god=%.2f, te_god_avg=%.2f',
                stats.test_opt:mean(),
                stats.test_opt_avg,
                stats.test_god:mean(),
                stats.test_god_avg)
        end
    end

    local function create_agent()

        -- Sizes
        local comm_size = 0

        if (opt.game_comm_bits > 0) and (opt.game_nagents > 1) then
            if opt.game_comm_limited then
                comm_size = opt.game_comm_bits
            else
                error('game_comm_limited is required')
            end
        end

        local action_aware_size = 0
        if opt.model_action_aware == 1 then
            action_aware_size = opt.game_action_space_total
        end


        -- Process inputs
        local model_input = nn.Sequential()
        model_input:add(nn.CAddTable(2))
        -- if opt.model_bn == 1 then model_input:add(nn.BatchNormalization(opt.model_rnn_size)) end

        local model_state = nn.Sequential()
        model_state:add(nn.LookupTable(2, opt.model_rnn_size))

        -- RNN
        local model_rnn
        if opt.model_rnn == 'lstm' then
            model_rnn = LSTM(opt.model_rnn_size,
                opt.model_rnn_size,
                opt.model_rnn_layers,
                opt.model_dropout,
                opt.model_bn == 1)
        elseif opt.model_rnn == 'gru' then
            model_rnn = GRU(opt.model_rnn_size,
                opt.model_rnn_size,
                opt.model_rnn_layers,
                opt.model_dropout)
        end

        -- use default initialization for convnet, but uniform -0.08 to .08 for RNN:
        -- double parens necessary
        for _, param in ipairs((model_rnn:parameters())) do
            param:uniform(-0.08, 0.08)
        end

        -- Output
        local model_out = nn.Sequential()
        if opt.model_dropout > 0 then model_out:add(nn.Dropout(opt.model_dropout)) end
        model_out:add(nn.Linear(opt.model_rnn_size, opt.model_rnn_size))
        model_out:add(nn.ReLU(true))
        model_out:add(nn.Linear(opt.model_rnn_size, opt.game_action_space_total))

        -- Construct Graph
        local in_state = nn.Identity()()
        local in_id = nn.Identity()()
        local in_rnn_state = nn.Identity()()

        local in_comm, in_action

        local in_all = {
            model_state(in_state),
            nn.LookupTable(opt.game_nagents, opt.model_rnn_size)(in_id)
        }

        -- Communication enabled
        if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
            in_comm = nn.Identity()()
            -- Process comm
            local model_comm = nn.Sequential()
            model_comm:add(nn.View(-1, comm_size))
            if opt.model_dial == 1 then
                if opt.model_comm_narrow == 1 then
                    model_comm:add(nn.Sigmoid())
                else
                    model_comm:add(nn.SoftMax())
                end
            end
            if opt.model_bn == 1 and opt.model_dial == 1 then
                model_comm:add(nn.BatchNormalization(comm_size))
            end
            model_comm:add(nn.Linear(comm_size, opt.model_rnn_size))
            if opt.model_comm_narrow == 1 then
                model_comm:add(nn.ReLU(true))
            end

            -- Process inputs node
            table.insert(in_all, model_comm(in_comm))
        end

        -- Last action enabled
        if opt.model_action_aware == 1 then
            in_action = nn.Identity()()

            -- Process action node (+1 for no-action at 0-step)
            if opt.model_dial == 0 then
                local in_action_aware = nn.CAddTable(2)({
                    nn.LookupTable(opt.game_action_space + 1, opt.model_rnn_size)(nn.SelectTable(1)(in_action)),
                    nn.LookupTable(opt.game_comm_bits + 1, opt.model_rnn_size)(nn.SelectTable(2)(in_action))
                })
                table.insert(in_all, in_action_aware)
            else
                table.insert(in_all, nn.LookupTable(action_aware_size + 1, opt.model_rnn_size)(in_action))
            end
        end

        -- Process inputs
        local proc_input = model_input(in_all)

        -- 2*n+1 rnn inputs
        local rnn_input = {}
        table.insert(rnn_input, proc_input)

        -- Restore state
        for i = 1, opt.model_rnn_states do
            table.insert(rnn_input, nn.SelectTable(i)(in_rnn_state))
        end

        local rnn_output = model_rnn(rnn_input)

        -- Split state and out
        local rnn_state = rnn_output
        local rnn_out = nn.SelectTable(opt.model_rnn_states)(rnn_output)

        -- Process out
        local proc_out = model_out(rnn_out)

        -- Create model
        local model_inputs = { in_state, in_id, in_rnn_state }
        local model_outputs = { rnn_state, proc_out }

        if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
            table.insert(model_inputs, in_comm)
        end

        if opt.model_action_aware == 1 then
            table.insert(model_inputs, in_action)
        end

        nngraph.annotateNodes()

        local model = nn.gModule(model_inputs, model_outputs)

        return model:type(opt.dtype)
    end

    -- Create model
    local agent = create_agent()
    local agent_target = agent:clone()

    -- Knowledge sharing
    if opt.model_know_share == 1 then
        exp.agent = util.cloneManyTimes(agent, opt.game_nagents * (opt.nsteps + 1))
        exp.agent_target = util.cloneManyTimes(agent_target, opt.game_nagents * (opt.nsteps + 1))
    else
        exp.agent = {}
        exp.agent_target = {}

        local agent_copies = util.copyManyTimes(agent, opt.game_nagents)
        local agent_target_copies = util.copyManyTimes(agent_target, opt.game_nagents)

        for i = 1, opt.game_nagents do
            local unrolled = util.cloneManyTimes(agent_copies[i], opt.nsteps + 1)
            local unrolled_target = util.cloneManyTimes(agent_target_copies[i], opt.nsteps + 1)
            for s = 1, opt.nsteps + 1 do
                exp.agent[exp.id(s, i)] = unrolled[s]
                exp.agent_target[exp.id(s, i)] = unrolled_target[s]
            end
        end
    end

    return exp
end