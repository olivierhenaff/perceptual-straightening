require 'image'

function char2str( t ) 

	local s = ''
	for i = 1, t:size(2) do s = s .. string.char( t[1][i] ) end 
	return s 

end

function split(s, delimiter)
    result = {};
    for match in (s..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match);
    end
    return result;
end

function loadSequence( condition, sequence )

	require 'mattorch'

	if model == 'data_yoon' then 

		local s = split( condition, '_' )

		local sequence_type = s[1] 
		local sequence_ecnt = s[2] 
		local sequence_name = s[3] .. '_' .. s[4] 
		local sequence_indx
		if     sequence_name == '04_EGOMOTION' then
			sequence_indx = 1 
		elseif sequence_name == '05_PRAIRIE' then 
			sequence_indx = 2
		elseif sequence_name == '06_DAM' then 
			sequence_indx = 3 
		end

		local loaddir = 'stimuli/stimuli/yoon/' .. sequence_type .. '_movie_frames_post.mat'
		local t = mattorch.load( loaddir )

		local y = t[ 'frames_' .. sequence_ecnt ]:select( 4, sequence_indx ) 

		return y 

	end

end

function computePixelfade( x ) 

	local nsmpl = x:size( 1 ) 
	local imgA = x[1]:clone()
	local imgB = x[nsmpl]:clone()
	local y = x:clone()

	for i = 1, nsmpl do 
		local t = (i-1)/(nsmpl-1)
		y[i]:copy( imgA ):mul( 1-t ):add( t, imgB ) 
	end 

	return y 

end

function computeDistances( x ) 

	local nsmpl = x:size( 1 ) 
	local d = torch.Tensor( nsmpl, nsmpl )

	for i = 1, nsmpl do 
		for j = 1, nsmpl do 
			d[i][j] = x[i]:dist( x[j] ) 
		end
	end

	return d 

end 