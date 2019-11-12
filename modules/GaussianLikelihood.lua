require 'nngraph'

function GaussianLikelihood( mu, sig )

	local inode = nn.Identity()()

	local likelihood = nn.Square()( nn.AddConstant( - mu )( inode ) )
	likelihood = nn.Exp()( nn.MulConstant( - 1 / ( 2 * sig^2 ) )( likelihood ) )
	likelihood = nn.MulConstant( 1 / ( sig*math.sqrt( 2*math.pi ) ) )( likelihood ) 

	local network = nn.gModule({ inode }, { likelihood })

	return network

end

function GaussianLogLikelihoodDiag( muSigDim ) 

	local muSigDim = muSigDim or 2 

	local inode = nn.Identity()()

	local z       = nn.SelectTable(1)( inode ) -- sample 
	local t       = nn.SelectTable(2)( inode )

	local muT     = nn.Select(muSigDim,1)( t ) --[ncond, nsmpl]
	local logSigT = nn.Select(muSigDim,2)( t )
	local    sigT = nn.Exp()( logSigT ) 

	local mse = nn.CSubTable()({ z, muT })
	mse       = nn.Square()( nn.CDivTable()({ mse, sigT }) ) 

	local logLikelihood = nn.CAddTable()({ mse, nn.MulConstant(2)( logSigT ) })
	logLikelihood = nn.AddConstant( math.log(2*math.pi) )( logLikelihood )  
	logLikelihood = nn.Sum( 2 )( logLikelihood ) 
	logLikelihood = nn.MulConstant( - 0.5 )( logLikelihood ) 

	local network = nn.gModule({ inode }, { logLikelihood })

	return network 

end

function GaussianLogLikelihoodEps( muSigDim ) 

	local muSigDim = muSigDim or 2 

	local inode = nn.Identity()()

	local e       = nn.SelectTable(1)( inode ) -- sample 
	local t       = nn.SelectTable(2)( inode )
	local logSigT = nn.Select(muSigDim,2)( t )

	local logLikelihood = nn.CAddTable()({ nn.Square()( e ), nn.MulConstant(2)( logSigT ) })
	logLikelihood = nn.AddConstant( math.log(2*math.pi) )( logLikelihood )  
	logLikelihood = nn.Sum( 2 )( logLikelihood ) 
	logLikelihood = nn.MulConstant( - 0.5 )( logLikelihood ) 

	local network = nn.gModule({ inode }, { logLikelihood })

	return network 

end



function testGaussianLikelihood()

	local mu  = 1 
	local sig = 2

	local likelihood = GaussianLikelihood( mu, sig )

	local x = torch.linspace( mu - 3*sig, mu + 3*sig, 1000 )
	local y = likelihood:updateOutput( x ) 

	local n = 10000 
	local z = torch.randn( n ):mul( sig ):add( mu ) 

	local ll = GaussianLogLikelihoodDiag()
	local  t = torch.Tensor( 1000, 2, 1 )
	t:select( 2, 1 ):fill( mu ) 
	t:select( 2, 2 ):fill( math.log( sig ) ) 
	local y2 = ll:updateOutput({ x:view(1000, 1), t }):exp()

	local plot = { {'likelihood', x,y,'-'}, {'exp(loglikelihood)', x,y2,'-'}, gnuplot.plotHist( z, 100, mu - 3*sig, mu + 3*sig ) }
	gnuplot.savePlot( 'gaussianLikelihood' .. mu .. '-' .. sig .. '.png', plot )

end

function testGaussianLogLikelihoodDiag()

	local mb    = 1 
	local nsmpl = 100 

	z = torch.randn( mb,    nsmpl ) 
	t = torch.randn( mb, 2, nsmpl )

	local ll = GaussianLogLikelihoodDiag()

	print( ll:updateOutput{ z, t } )


end