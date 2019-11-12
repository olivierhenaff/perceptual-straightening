require 'nngraph'

function pCorrectLapsed( ncond, nsmpl )

	local inode = nn.Identity()()

	local lapse = nn.SelectTable(1)( inode ) 
	local pABX  = nn.SelectTable(2)( inode ) 

	local oneMinusLapse = nn.AddConstant( 1 )( nn.MulConstant( -1 )( lapse ) )
	oneMinusLapse = nn.Replicate( ncond*nsmpl*nsmpl, 1 )( oneMinusLapse ) 

	local lapseOverTwo = nn.MulConstant( 0.5 )( lapse ) 
	lapseOverTwo = nn.Replicate( ncond*nsmpl*nsmpl, 1 )( lapseOverTwo )

	local pLapsed = nn.CAddTable()({nn.CMulTable()({pABX, oneMinusLapse}),lapseOverTwo})

	local network = nn.gModule({ inode }, { pLapsed })

	return network 

end