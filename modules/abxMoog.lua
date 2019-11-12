function abxMoog( mb, dim, nsmpl )

	local inode = nn.Identity()()

	local d = nn.SelectTable( 1 )( inode ):annotate{ name = 'distances'  }
	local t = nn.SelectTable( 2 )( inode ):annotate{ name = 'theta1' }
	local a = nn.SelectTable( 3 )( inode ):annotate{ name = 'acc'   }

	t = nn.MulConstant( math.pi )( t ):annotate{ name = 'thetaPi' } 

	local z = accCurvatureToZ( mb, dim, nsmpl )({ d, t, a }):annotate{ name = 'trajectory' }

	pABX = nn.View( mb * dim, nsmpl )( z )
	pABX = Differences( nsmpl )( pABX ):annotate{ name = 'differences' }
	pABX = nn.View( mb , dim, nsmpl, nsmpl )( pABX ) 
	pABX = nn.Square()( pABX ) 
	pABX = nn.Sum(2)( pABX ):annotate{ name = 'squared distances' } 
	pABX = nn.Sqrt()( pABX ):annotate{ name = 'distances' } 
	pABX = ABXpCorrect()( pABX ):annotate{ name = 'pABX' }

	local network = nn.gModule({ inode }, { pABX })

	return network 

end