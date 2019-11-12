function lowerTriangle( dim ) 

	local cmul = nn.CMul( 1, dim, dim ) 
	cmul.weight:copy( torch.tril( torch.ones(dim,dim), -1 ) )

	local cadd = nn.CAdd( 1, dim, dim )
	cadd.bias:copy( torch.eye( dim ) ) 

	local inode = nn.Identity()()
	local output = cadd( cmul( inode ) )

	local network = nn.gModule({inode}, {output})

	return network 

end