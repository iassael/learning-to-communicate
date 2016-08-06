--[[

    Based on the code of Brendan Shillingford [bitbucket.org/bshillingford/nnob](https://bitbucket.org/bshillingford/nnob).

    Argument type checker for keyword arguments, i.e. arguments
    specified as a key-value table to constructors/functions.

    ## Valid typespecs:
     - `"number"`
     - `"string"`
     - `"boolean"`
     - `"function"`
     - `"table"`
     - `"tensor"`
     - torch class (not a string) like nn.Module
     - `"class:x"`
        - e.g. x=`nn.Module`; can be comma-separated to OR them
        - specific `torch.*Tensor`
        - specific `torch.*Storage`
     - `"int"`: number that is integer
     - `"int-pos"`: integer, and `> 0`
     - `"int-nonneg"`: integer, and `>= 0`
--]]
require 'torch'
local math = require 'math'

local function assert_type(val, typespec, argname)
    local typename = torch.typename(val) or type(val)
    -- handles number, boolean, string, table; but passes through if needed
    if typespec == typename then return true end

    -- try to parse, nil if no match
    local classnames = type(typespec) == string and string.match(typespec, 'class: *(.*)')

    -- isTypeOf for table-typed specs (see below for class:x version)
    if type(typespec) == 'table' and typespec.__typename then
        if torch.isTypeOf(val, typespec) then
            return true
        else
            error(string.format('argument %s should be instance of %s, but is type %s',
                argname, typespec.__typename, typename))
        end
    elseif typespec == 'tensor' then
        if torch.isTensor(val) then return true end
    elseif classnames then
        for _, classname in pairs(string.split(classnames, ' *, *')) do
            if torch.isTypeOf(val, classname) then return true end
        end
    elseif typespec == 'int' or typespec == 'integer' then
        if math.floor(val) == val then return true end
    elseif typespec == 'int-pos' then
        if val > 0 and math.floor(val) == val then return true end
    elseif typespec == 'int-nonneg' then
        if val >= 0 and math.floor(val) == val then return true end
    else
        error('invalid type spec (' .. tostring(typespec) .. ') for arg ' .. argname)
    end
    error(string.format('argument %s must be of type %s, given type %s',
        argname, typespec, typename))
end

return function(args, settings)
    local result = {}
    local unprocessed = {}

    if not args then
        args = {}
    end

    if type(args) ~= 'table' then
        error('args must be non-nil and must be a table')
    end

    for k, _ in pairs(args) do
        unprocessed[k] = true
    end

    -- Use ipairs, so we skip named settings
    for _, setting in ipairs(settings) do
        -- allow name to either be the only non-named element
        -- e.g. {'name', type='...'}, or named
        local name = setting.name or setting[1]

        -- get value or default
        local val
        if args[name] ~= nil then
            val = args[name]
        elseif setting.default ~= nil then
            val = setting.default
        elseif not setting.optional then
            error('required argument: ' .. name)
        end
        -- check types
        if val ~= nil and not setting.optional and setting.type ~= nil then
            assert_type(val, setting.type, name)
        end

        result[name] = val
        unprocessed[name] = nil
    end

    if settings.ignore_extras then
        for _, name in pairs(unprocessed) do
            result[name] = args[name]
        end
    elseif #unprocessed > 0 then
        error('extra unprocessed arguments: '
                .. table.concat(unprocessed, ', '))
    end
    return result
end
