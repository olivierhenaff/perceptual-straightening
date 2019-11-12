require 'nngraph'

require 'modules/cdfNormal'

-------------------------------------------------------------
--- percentage correct as a function of distances -----------
-------------------------------------------------------------

function ABXpCorrect()

	local inode = nn.Identity()()

	local phi11 = nn.cdfNormal()( nn.MulConstant(   1 / math.sqrt( 2 ) )( inode ) )
	local phi12 = nn.cdfNormal()( nn.MulConstant(   1 / 2 )( inode ) )
	local phi21 = nn.cdfNormal()( nn.MulConstant( - 1 / math.sqrt( 2 ) )( inode ) )
	local phi22 = nn.cdfNormal()( nn.MulConstant( - 1 / 2 )( inode ) )

	local pABX = nn.CAddTable()({nn.CMulTable()({phi11, phi12}), nn.CMulTable()({phi21, phi22})})

	local network = nn.gModule({ inode }, { pABX })

	return network

end

function visualizeABX()

	local x = torch.linspace( 0, 5, 1000 )
	local p = ABXpCorrect() 
	local y = p:updateOutput( x ) 

	require 'common-torch/utilities/gnuplot'

	gnuplot.savePlot( 'abx.png', {x, y, '-'}, 'distance', 'proportion correct', {'','',0.5,1} )

end
-- visualizeABX()

function linearStepGMod( input, x1, x2, y1, y2 )

	print( 'gmod', 1 )
	local output = nn.MulConstant( 2/(x2-x1) )( input ) 
	print( 'gmod', 2 )
	output = nn.AddConstant(-1-2*x1/(x2-x1))( output ) 
	print( 'gmod', 3 )
	output = nn.HardTanh()( output ) 
	print( 'gmod', 4 )
	output = nn.AddConstant(1)( output ) 
	print( 'gmod', 5 )
	output = nn.MulConstant( (y2-y1)/2 )( output ) 
	print( 'gmod', 6 )

	return output 

end

function linearStep( x1, x2, y1, y2 )


	-- error('stop here')


	-- local net = nn.Sequential()
	-- 	:add( nn.MulConstant( 2/(x2-x1) ) )
	-- 	:add( nn.AddConstant(-1-2*x1/(x2-x1)) )
	-- 	:add( nn.HardTanh() )
	-- 	:add( nn.AddConstant(1) )
	-- 	:add( nn.MulConstant( (y2-y1)/2 ) )

	-- print( 'linear', 1 )
	local net = nn.Sequential()
	-- print( 'linear', 2 )
	net:add( nn.MulConstant( 2/(x2-x1) ) )
	-- print( 'linear', 3 )
	net:add( nn.AddConstant(-1-2*x1/(x2-x1)) )
	-- print( 'linear', 4 )
	net:add( nn.HardTanh() )
	-- print( 'linear', 5 )
	net:add( nn.AddConstant(1) )
	-- print( 'linear', 6 )
	net:add( nn.MulConstant( (y2-y1)/2 ) )
	-- print( 'linear', 7 )

	-- error('stop here')

	return net 

end

function piecewiseLinearApproxGMod( x, y ) 

	local inode = nn.Identity()()
	local steps = {} 

	-- error('stop here')

	for i = 1, x:size(1)-1 do 

		print( i, 1 ) 

		local step = linearStepGMod( inode, x[i], x[i+1], y[i], y[i+1] )

		print( i, 2 ) 

		table.insert( steps, step)

		print( i, 3 ) 

	end

	error('stop here')


	local output = nn.CAddTable()( steps ) 
	output = nn.AddConstant( y[1] )( output ) 

	local network = nn.gModule( {inode}, {output} )

	return network

end

function piecewiseLinearApprox( x, y ) 

	local cat = nn.ConcatTable() 

	for i = 1, x:size(1)-1 do 

		local step = linearStep( x[i], x[i+1], y[i], y[i+1] )

		cat:add( step ) 

	end 

	local net = nn.Sequential()
		:add( cat )
		:add( nn.CAddTable() )
		:add( nn.AddConstant( y[1] ) ) 

	return net 

end

function ABXpCorrectInverse( m, n )  

	local n = n or 1000
	local m = m or 10 	

	local p = ABXpCorrect()
	local x = torch.linspace(0, m, n)
	local y = p:updateOutput( x )
	local pInv = piecewiseLinearApprox( y, x ) 

	return pInv 

end 


function fitABXpCorrectInverse()

	require 'common-torch/utilities/gnuplot'

	local n = 1000
	local m = 10 
	local p = ABXpCorrect()
	local x = torch.linspace(0, m, n)
	local y = p:updateOutput( x ):clone()

	local pApprox = piecewiseLinearApprox( x, y )
	local z = pApprox:updateOutput( x ):clone()

	local xTest = torch.rand(n):mul(m)
	local y1 = p:updateOutput( xTest ) 
	local y2 = pApprox:updateOutput( xTest ) 

	print('difference (true, piecewiseLinearApprox) = ', y1:dist(y2) ) 


	gnuplot.savePlot( 'abx.png', {{x,y,'-'}, {x,z,'-'}})

	local pInv = ABXpCorrectInverse( m, n ) --piecewiseLinearApprox( y, x ) 
	xHat = pInv:updateOutput( y ) 

	local plot = {{y,x,'-'}, {y,xHat,'-'}}
	gnuplot.savePlot( 'abxInv.png', plot)

end
--fitABXpCorrectInverse()
