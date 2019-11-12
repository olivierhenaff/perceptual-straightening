require 'randomkit'


function binaryToInd( x )

	local t = {} 

	for i = 1, x:size(1) do
		if x[i] == 1 then table.insert( t, i ) end 
	end

	return torch.LongTensor( t ) 

end

function filterDimension( x, dimensionMask, dim )

	local ind = binaryToInd( dimensionMask )
	return x:index( dim, ind )

end


function completeData( data ) 

	data.all = data.right:clone():add( data.wrong )
	local d = data.all:dim() 
	local oTrials = data.all:eq( 0 ):double()
	data.pCorrect = data.right:clone():cdiv( data.all:clone():add( oTrials ) )
	for i = 1, data.pCorrect:size(2) do 
		data.pCorrect:narrow(d-1,i,1):narrow(d,i,1):fill(0.5)
	end

end

function refineData( data, max_deviation, min_deviation) 

	local min_deviation = min_deviation or -math.huge 
	local max_deviation = max_deviation or  math.huge 

	local raw = data.raw

	if raw then 

		raw[1]:fill(1) 
		if model == 'data_corey' then 
			raw[2]:add( -7 )
			raw[3]:add( -7 ) 
			raw[4]:fill(0)
			raw[5]:fill(0)
			for i = 2, 3 do 
				local inds = raw[i]
				local filt = inds:le(11):cmul( inds:ge(1) )
				raw = filterDimension(raw, filt, 2)
			end
		end
		local ncond, nsmpl = raw[1]:max(), raw[2]:max() 
		data.right = torch.zeros( ncond, nsmpl, nsmpl )
		data.wrong = torch.zeros( ncond, nsmpl, nsmpl )

		local cond, indi, indj, devi, crct 

		cond, indi, indj, devi, crct  = raw[1], raw[2], raw[3], raw[4], raw[6]:eq( raw[7] )

		for t = 1, raw:size(2) do 

			if min_deviation < devi[t] and devi[t] < max_deviation and indi[t] > 0 and indj[t] > 0 then

				if crct[t] == 1 then 
					data.right[cond[t]][indi[t]][indj[t]] = data.right[cond[t]][indi[t]][indj[t]] + 1 
				else
					data.wrong[cond[t]][indi[t]][indj[t]] = data.wrong[cond[t]][indi[t]][indj[t]] + 1 
				end

			end

		end

		data.raw = nil

		completeData( data )

	end
end


function loadDataRec( sequence, name, tags, max_deviation, raw, onlyNatural )

	local dir 
	if     model == 'data_yoon' then 
		dir = 'data/t7/yoon/' 
		dir = dir .. name .. '_' .. sequence .. '.t7'
	elseif model == 'data_corey' then 
		dir = 'data/t7/corey/' 
		dir = dir .. name .. '_' .. sequence .. '.t7'
	end

	local data = { sequence = sequence, name = name }

	local raw
	if paths.filep( dir ) then

		raw = torch.load(dir)
		data.raw = raw 

	end 

	refineData( data, max_deviation, min_deviation ) 

	return data 

end