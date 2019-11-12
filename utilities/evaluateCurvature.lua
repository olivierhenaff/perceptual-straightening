function computeSpeed( x ) 

	local nsmpl = x:size( 1 ) 
	local v = x:narrow( 1, 2, nsmpl-1 ):clone():add( -1, x:narrow( 1, 1, nsmpl-1 ) )
	local d = torch.Tensor( nsmpl-1 )
	for i = 1, nsmpl-1 do 

		d[i] = v[i]:norm() + 1e-8

		v[i]:div( d[i] ) 

	end

	return v, d 

end 

function computeDistCurvature( x ) 

	local nsmpl = x:size( 1 ) 
	local v, d  = computeSpeed( x ) 
	local c     = torch.Tensor( nsmpl - 2 )
	local aux   = torch.Tensor( v[1]:size() ):typeAs( v )

	for i = 1, nsmpl-2 do

		aux:copy( v[i] ):cmul( v[i+1] )
		local cos = aux:sum(); cos = math.min( cos, 1 ); cos = math.max( cos, -1 ) 
		c[i] = math.acos( cos ) 
		-- print( i, 180 * c[i] / math.pi ); 
	end 

	local e = x[1]:dist( x[nsmpl] ) / ( nsmpl-1 ) 

	return d, c, e  

end 


function computeCurvaturePointwise( x ) 

	if x then 

		local nsmpl = x:size( 1 ) 
		local v, _  = computeSpeed( x ) 
		local curv  = torch.Tensor( nsmpl-2 )
		local aux   = torch.Tensor( v[1]:size() ):typeAs( v )

		for i = 1, nsmpl-2 do

			aux:copy( v[i] ):cmul( v[i+1] )
			local cos = aux:sum(); cos = math.min( cos, 1 ); cos = math.max( cos, -1 ) 
			curv[i] = math.acos( cos ) 

		end 

		return curv 

	else

		return torch.Tensor(1):fill(0/0)

	end

end 

function computeCurvature( x ) 

	local curv = computeCurvaturePointwise( x ) 

	return curv:sum()

end

function computeAverageCurvature( x )

	local totalCurvature = computeCurvature( x )

	return totalCurvature / ( (x:size(1)-2) * math.pi )

end

function computeDistCurvAcc( x )

	local nsmpl = x:size(1)
	local dim   = x:size(2) 

	local v, _ = computeSpeed( x ) 
	local d, c = computeDistCurvature( x )

	local z0 = x[1]:clone() 
	local v0 = x[2]:clone():add( -1, x[1] ) 
	local a  = torch.Tensor( dim, nsmpl-2 ) 
	for i = nsmpl-2, 1, -1 do a:select( 2, i ):copy( v[i+1] ):add( -math.cos(c[i]), v[i] ) end

	return d, c, a 

end