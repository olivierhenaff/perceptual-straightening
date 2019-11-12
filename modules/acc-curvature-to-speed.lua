require 'nngraph'

require 'modules/constTensor'
require 'modules/Cos'

function grahamSchmidt( ncond, dim )

	local inode = nn.Identity()()

	local a = nn.SelectTable(1)( inode ) -- ncond, dim, 1 
	local v = nn.SelectTable(2)( inode ) -- ncond, dim, 1  

	local dot = nn.Sum(2)( nn.CMulTable()({ a, v }) )
	dot = nn.View( ncond, 1, 1 )( dot )
	dot = nn.Replicate( dim, 2 )( dot )

	local aHat = nn.CSubTable()({ a, nn.CMulTable()({dot, v}) })

	local norm = nn.Sqrt()( nn.Sum(2)( nn.Square()( aHat ) ) ) 
	norm = nn.View( ncond, 1, 1 )( norm )
	norm = nn.Replicate( dim, 2 )( norm ) 

	aHat = nn.CDivTable()({ aHat, norm })

	local network = nn.gModule( {inode}, {aHat} )

	return network 

end

function accCurvatureToVhat( ncond, dim, nsmpl )

	local inode = nn.Identity()()

	local t = nn.SelectTable(1)( inode ):annotate{ name = 'theta' } -- ncond,   1, nsmpl-2
	local a = nn.SelectTable(2)( inode ):annotate{ name = 'accel' } -- ncond, dim, nsmpl-2

	local e1 = torch.Tensor( ncond, dim, 1 ):zero(); e1:select(2,1):fill(1)
	local vHat = { nn.ConstTensor(e1)( t ):annotate{ name = 'vHat'..1 } }

	local cosT = nn.Replicate( dim, 2 )( nn.Cos()( t ) ):annotate{ name = 'cosTheta' }
	local sinT = nn.Replicate( dim, 2 )( nn.Sin()( t ) ):annotate{ name = 'sinTheta' }

	for i = 1, nsmpl-2 do

		local aHat = grahamSchmidt( ncond, dim )({ nn.Narrow( 3, i, 1 )( a ), vHat[i] })

		local cI = nn.Narrow( 3, i, 1 )( cosT ):annotate{ name = 'cosTheta'..i }
		local sI = nn.Narrow( 3, i, 1 )( sinT ):annotate{ name = 'sinTheta'..i } 

		local cosTvHat = nn.CMulTable()({ cI, vHat[i] })
		local sinTaHat = nn.CMulTable()({ sI, aHat    })

		vHat[i+1] = nn.CAddTable()({ cosTvHat, sinTaHat }):annotate{ name = 'vHat'..i+1 }

	end

	vHat = nn.JoinTable( 3 )( vHat )

	local network = nn.gModule( {inode}, {vHat} )

	return network 

end 



function accCurvatureToZ( ncond, dim, nsmpl )

	local inode = nn.Identity()()

	local d = nn.SelectTable(1)( inode ) -- ncond,   1, nsmpl-1
	local t = nn.SelectTable(2)( inode ) 
	local a = nn.SelectTable(3)( inode ) 

	local vHat = accCurvatureToVhat( ncond, dim, nsmpl )({ t, a })

	local d = nn.Replicate( dim, 2 )( d ) 
	local v = nn.CMulTable()({ d, vHat } ) -- ncond, dim, nsmpl-1

	local z0 = torch.Tensor( ncond, dim, 1 ):zero()
	local z = { nn.ConstTensor(z0)( v ) } 

	for i = 2, nsmpl do 

		z[i] = nn.CAddTable()({ z[i-1], nn.Narrow( 3, i-1, 1)( v ) })

	end 

	z = nn.JoinTable( 3 )( z ) 

	local network = nn.gModule( {inode}, {z} )

	return network

end 



function testGrahamSchmidt() 

	local ncond = 10
	local dim   = 5 

	local a = torch.randn( ncond, dim, 1 )
	local v = torch.randn( ncond, dim, 1 )

	for i = 1, ncond do v[i]:div( v[i]:norm() ) end 

	local gs   = grahamSchmidt( ncond, dim ) 
	local aHat = gs:updateOutput({ a, v })

	for i = 1, ncond do 

		print( aHat[i]:norm(), v:dot( aHat ) ) 

	end	

end
-- testGrahamSchmidt() 


