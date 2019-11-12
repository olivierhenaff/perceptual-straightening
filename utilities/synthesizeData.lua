function synthesizePerceptualData( p, all, oracle )

	data = {right    = torch.Tensor( p:size() ),
			wrong    = torch.Tensor( p:size() ),
			all      = torch.Tensor( p:size() ), 
			pCorrect = torch.Tensor( p:size() )}

	data.all:copy( all ) 

	if oracle then

		data.right:copy( p ):cmul( data.all )

	else

		if dataMultiplier then data.all:mul( dataMultiplier ) end 

		local erroneous = true 
		local count     = 0 

		while erroneous do 

			count = count + 1 
			randomkit.binomial( data.right, data.all, p )
			if data.right:gt( data.all ):sum() == 0 then erroneous = false end 

		end 

		print( data.right:gt( data.all ):sum(), 'instances of right being greater than all, after ', count, 'tries' ) 

	end
	data.wrong:copy( data.all ):add( -1, data.right ) 

	local oTrials = data.all:eq(0):double()
	data.pCorrect = data.right:clone():cdiv( data.all:clone():add( oTrials ) )
	for i = 1, data.pCorrect:size(2) do data.pCorrect[i][i] = 0.5 end

	print( 'distance true, sampled pCorrect', p:dist( data.pCorrect ) ) 

	return data 

end