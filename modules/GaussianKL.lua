require 'nngraph'

function GaussianKLstandard( ncond, muSigDim )

	local inode = nn.Identity()()

	local mean   = nn.Select( muSigDim, 1 )( inode )
	local logStd = nn.Select( muSigDim, 2 )( inode )

	local mean2 = nn.Square()( mean ) 
	local std2  = nn.Square()( nn.Exp()( logStd ) )

	local kl = nn.CAddTable()({nn.MulConstant(-2)(logStd), mean2, std2})

	kl = nn.AddConstant(-1)( kl ) 
	kl = nn.View(ncond, -1)( kl ) 
	kl = nn.Sum(2)( kl )
	kl = nn.MulConstant( 0.5 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

function GaussianKLdiagonalLearnedMuStd( ncond, muSigDim, prior ) 

	local muSigDim = muSigDim or 1 
	
	local inode = nn.Identity()()
	local terms = {} 

	local z       = nn.SelectTable(1)( inode )
	local logSigZ = nn.Select(muSigDim,2)( z )

	local t       = nn.SelectTable(2)( inode )
	local logSigT = nn.Select(muSigDim,2)( t )

	local twoLogSigT = nn.MulConstant( 2 )( logSigT )
	local sigThetaInv2 = nn.Exp()( nn.MulConstant(-1)(twoLogSigT) ) 

	local sigZ2 = nn.Exp()( nn.MulConstant(2)( logSigZ ) ) 
	local stdRatio = nn.CMulTable()({sigZ2, sigThetaInv2})

	local minusTwoLogSigZ = nn.MulConstant(-2)( logSigZ )

	local terms = {twoLogSigT, minusTwoLogSigZ, stdRatio}

	if prior == 'scalar' then 

		local muZ  = nn.Select(muSigDim,1)( z )
		local muT  = nn.Select(muSigDim,1)( t ) --ncond, nsmpl
		local muZ2 = nn.Square()( nn.CSubTable()({muZ, muT}) ) 
		local differences = nn.CMulTable()({muZ2, sigThetaInv2})
		table.insert( terms, differences ) 

	elseif prior ~= 'full' then 

		error('unknown prior distribution: ' .. prior )

	end

	local kl = nn.CAddTable()( terms ) 
	kl = nn.AddConstant(-1)( kl ) 
	kl = nn.View(ncond, -1)( kl ) 
	kl = nn.Sum(2)( kl )
	kl = nn.MulConstant( 0.5 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

function GaussianKLdiagonalLearnedMuStdFull( ncond, muSigDim, n, prior, rotateFirst ) 

	local muSigDim = muSigDim or 1 
	
	local inode = nn.Identity()()

	local z        = nn.SelectTable(1)( inode )
	local logSigZ  = nn.Select(muSigDim,2)( z )

	local t        = nn.SelectTable(2)( inode )
	local logSigT  = nn.Select(muSigDim,2)( t )

	local      twoLogSigT = nn.MulConstant( 2 )( logSigT )
	local minusTwoLogSigZ = nn.MulConstant(-2 )( logSigZ )
	local terms = {twoLogSigT, minusTwoLogSigZ}

	local sigTi = nn.Exp()( nn.MulConstant(-1)( logSigT ) ) 
	local sigZ  = nn.Exp()(                     logSigZ ) 

	if     prior == 'scalar' then 

	 	local muZ     = nn.Select(muSigDim,1)( z )
		local muT     = nn.Select(muSigDim,1)( t ) --[ncond, nsmpl]
		local deltaMu = nn.CSubTable()({muZ, muT}) 
		local deltaSc = nn.CMulTable()({deltaMu, sigTi})
		local mse     = nn.Square()( deltaSc ) 
		table.insert( terms, mse ) 

	elseif prior ~= 'full' then 

		print( prior ) 
		error('unknown prior distribution' )

	end

	sigZ  = nn.Replicate( n, rotateFirst and 3 or 2 )( sigZ  ) 
	sigTi = nn.Replicate( n, 3 )( sigTi )
	local rotation = nn.SelectTable(3)( inode )
	local trace = nn.Square()( nn.CMulTable()({sigZ, sigTi, rotation}) )
	trace = nn.Sum( 2 )( trace ) 
	table.insert( terms, trace ) 

	local kl = nn.CAddTable()( terms ) 
	kl = nn.AddConstant(-1)( kl ) 
	kl = nn.View(ncond, -1)( kl ) 
	kl = nn.Sum(2)( kl )
	kl = nn.MulConstant( 0.5 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

----------------------------------------------------------------------------------------
--- tests ------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

function testGaussianKLdiagonalLearnedMuStdFull() -- OK: analytical and sampled solutions match. 

	nngraph.setDebug( true ) 

	local mb    = 1 
	local nsmpl = 11 
	local kl1   = GaussianKLdiagonalLearnedMuStdFull( mb, 2, nsmpl, 'scalar' )

	local z = torch.randn( mb, 2, nsmpl )
	local t = torch.randn( mb, 2, nsmpl ) 
	local r = torch.randn( mb, nsmpl, nsmpl ) 

	local muZ     = z:select( 2, 1 )
	local muT     = t:select( 2, 1 ) 
	local logSigZ = z:select( 2, 2 )
	local logSigT = t:select( 2, 2 ) 

	for i = 1, mb do r[i]:cmul( torch.tril( torch.ones(nsmpl,nsmpl), -1 ) ):add( torch.eye( nsmpl ) ) end 

	local o1 = kl1:updateOutput({ z, t, r }):squeeze()

	local sigZ = logSigZ:clone():exp() 
	local sigT = logSigT:clone():exp() 

	local nSamples = 100000
	local e = torch.Tensor( nsmpl )
	local z = torch.Tensor( nsmpl ) 
	local kl2 = torch.Tensor( nSamples ) 
	local de = e:clone()

	for i = 1, nSamples do 

		e:randn( nsmpl ) 
		z:mv( r:squeeze(), de:copy(e):cmul( sigZ ) ):add( muZ ) 

		kl2[i] = logSigT:sum() - logSigZ:sum() - ( e:norm()^2 )/2 + ( z:add( -1, muT ):cdiv( sigT ):norm()^2 )/2 

	end

	o2 = kl2:mean() 

	print( o1, o2, math.abs(o1-o2) ) 

end 
-- testGaussianKLdiagonalLearnedMuStdFull()

function testGaussianKLstandardVsLearned() 

	local mb = 11
	local muSigDim = 2 
	local nsmpl = 1 

	local posterior = torch.randn( mb, 2, nsmpl )
	local prior     = posterior:clone():zero()

	local kl0 = GaussianKLstandard( mb, muSigDim )
	local kl1 = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )

	kl0:updateOutput(  posterior )
	kl1:updateOutput({ posterior, prior })

	print( kl0.output:dist( kl1.output ) ) 

end 

----------------------------------------------------------------------------------------
-- sample-based KL modules for unit testing analytical ones ----------------------------
----------------------------------------------------------------------------------------

require 'modules/SampleGaussian'
require 'modules/GaussianLikelihood'

function GaussianKLsampled( mu, sig, nsamples )

	local inode = nn.Identity()()

	local samples = nn.Replicate( nsamples, 2 )( inode ) 
	samples = SampleGaussianMuLogStd()( samples ) 

	local q = GaussianLikelihood( mu, sig )( samples )
	local p = GaussianLikelihood(  0,   1 )( samples )

	local kl = nn.Log()( nn.CDivTable()({q,p}) )
	kl = nn.Mean( 1 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

function testGaussianKLstandard()

	local KL1 = GaussianKLstandard()

	for i = 1, 100 do 

		local muLogSig = torch.randn( 2, 1 ) 
		local  mu = 		  muLogSig[1]:squeeze()
		local sig = math.exp( muLogSig[2]:squeeze() ) 

		local KL2 = GaussianKLsampled( mu, sig, 100000 )

		local kl1 = KL1:updateOutput( muLogSig ) 
		local kl2 = KL2:updateOutput( muLogSig ) 

		print( kl1:dist( kl2 ) ) 

	end

end 
-- testGaussianKLstandard()



function testGaussianPosteriorFull() 

	local mb = 1 
	local n  = 11 

	local kl1 = GaussianKL_priorStandard_posteriorFull( mb, 2 )
	local kl2 = GaussianKLdiagonalLearnedMuStdFull(     mb, 2, n, 'scalar' )

	local z = torch.randn( mb, 2, n ) 
	local t = torch.randn( mb, 2, n ) 
	local r = torch.randn( mb, n, n )

	local o1 = kl1:updateOutput{ z,    r }
	local o2 = kl2:updateOutput{ z, t, r }
	print( o1:squeeze(), o2:squeeze() ) 

	t:zero()

	local o1 = kl1:updateOutput{ z,    r }
	local o2 = kl2:updateOutput{ z, t, r }
	print( o1:squeeze(), o2:squeeze() ) 

end
-- testGaussianPosteriorFull()






