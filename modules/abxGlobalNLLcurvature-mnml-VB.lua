require 'cephes'
require 'randomkit'
require 'nngraph'

require 'modules/cdfNormal'
require 'modules/differences'
require 'modules/SampleGaussian'
require 'modules/GaussianKL'
require 'modules/acc-curvature-to-speed'
require 'modules/lapse'
require 'modules/segment'

-------------------------------------------------------------
--- likelihood of data as a function of distances -----------
-------------------------------------------------------------

local likelihood, parent = torch.class('nn.ABX_NLL_MNML_polar_VB', 'nn.Module')

function likelihood:__init( data, dim, mb, init, sampleMode )

	parent.__init(self)

	local losses = {}

	local nsmpl = data[1]:size( 2 )
	local nZ    = (nsmpl-1) + (nsmpl-2) + (nsmpl-2)*dim + lapseDim - 1

	local wcorrt = nn.CMul( data[1]:size() ) 
	local wwrong = nn.CMul( data[2]:size() ) 
	local wcombi = nn.Add(  data[1]:size() ) 

	self.ncorrt = wcorrt.weight:zero() 
	self.nwrong = wwrong.weight:zero() 
	self.combin = wcombi.bias:zero()

	local inode = nn.Identity()()

	local params = Segment( init, 2 )( inode ):annotate{ name = 'segment'  }

	local      d   = nn.SelectTable( 1 )( params ):annotate{ name =  'preDistParams' }
	local      t   = nn.SelectTable( 2 )( params ):annotate{ name = 'preThetaParams' }
	local      a   = nn.SelectTable( 3 )( params ):annotate{ name =   'preAccParams' }
	local priorD   = nn.SelectTable( 4 )( params ):annotate{ name = 'priorDist'  }
	local priorT   = nn.SelectTable( 5 )( params ):annotate{ name = 'priorTheta' }
	local priorA   = nn.SelectTable( 6 )( params ):annotate{ name = 'priorAcc' }
	local      l   = nn.SelectTable( 7 )( params ):annotate{ name = 'preLapseParams' }

	--- preprocess params 

	if thetaTransfer == 'SigmoidIdentity' then 
		local  muT = nn.Narrow( 2, 1, 1 )( t ) 
		local sigT = nn.Narrow( 2, 2, 1 )( t ) 

		muT = nn.Sigmoid()(              muT ):annotate{ name = 'thetaParams01' }
		muT = nn.MulConstant( math.pi )( muT ):annotate{ name = 'thetaParamsPi' }
		t = nn.JoinTable( 2 )( { muT, sigT } )
		
	end

	a      = nn.View( mb, 2, (nsmpl-2)*dim )( a )

	priorD = nn.Replicate( nsmpl-1, 3 )( priorD )
	priorT = nn.Replicate( nsmpl-2, 3 )( priorT )
	priorA = nn.Replicate( nsmpl-2, 3 )( priorA )
	priorA = nn.MulConstant(1)( priorA ) 
	priorA = nn.View( mb, 2, (nsmpl-2)*dim )( priorA )

	if accPrior == 'standard' then 

		priorA = nn.MulConstant(0)( priorA )

	elseif accPrior:find('zero') then 

		local  muA = nn.Narrow( 2, 1, 1 )( priorA ) 
		local sigA = nn.Narrow( 2, 2, 1 )( priorA ) 

		if accPrior == 'zeroMean' then  muA = nn.MulConstant(0)(  muA ) end 
		if accPrior == 'zeroSig'  then sigA = nn.MulConstant(0)( sigA ) end 

		priorA = nn.JoinTable( 2 )( { muA, sigA } )

	end

	local paramsPost  = {      d,      t,      a } 
	local paramsPrior = { priorD, priorT, priorA }

	if lapseDim > 1 then 

		l = nn.View( mb, 2, 1 )( l ) 
		local priorL = nn.MulConstant(0)( l ):annotate{ name = 'priorLapse' }

		table.insert( paramsPost ,      l ) 
		table.insert( paramsPrior, priorL )

	end

	--- compute kl divergence 

	local paramsPost  = nn.JoinTable( 3 )( paramsPost  )
	local paramsPrior = nn.JoinTable( 3 )( paramsPrior )
	local kl, rotation 
	if     zPost == 'full' then 

		rotation = nn.SelectTable( 8 )( params )
		rotation = lowerTriangle( nZ )( rotation ) 
		kl = GaussianKLdiagonalLearnedMuStdFull( mb, 2, nZ, 'scalar', rotateFirst )({ paramsPost, paramsPrior, rotation })

	elseif zPost == 'diagonal' then 

		kl = GaussianKLdiagonalLearnedMuStd(     mb, 2,     'scalar' )({ paramsPost, paramsPrior })

	end
	table.insert( losses, kl ) 

	--- sample from posterior 

	if sampleMode then 

		if sampleMode.all or sampleMode.d then d = setLogStdToNegInf( 2 )( d ) end 
		if sampleMode.all or sampleMode.t then t = setLogStdToNegInf( 2 )( t ) end 
		if sampleMode.all or sampleMode.a then a = setLogStdToNegInf( 2 )( a ) end

		if lapseDim > 1 then 

			if sampleMode.all or sampleMode.l then l = setLogStdToNegInf( 2 )( l ) end 

		end

 	end

	local paramsPost  = { d, t, a }
	if lapseDim > 1 then table.insert( paramsPost, l ) end 
 	paramsPost = nn.JoinTable( 3 )( paramsPost )

 	local samplePost

 	if zPost == 'full' then 
	 	if rotateFirst then 
	 		samplePost = SampleGaussianRotationLogStdMu( 2, mb, nZ )( {paramsPost, rotation } ):annotate{ name = 'distThetaAccLapseSample' }
	 	else
			samplePost = SampleGaussianMuLogStdRotation( 2, mb, nZ )( {paramsPost, rotation } ):annotate{ name = 'distThetaAccLapseSample' }
	 	end
 	elseif zPost == 'diagonal' then 
 		samplePost = SampleGaussianMuLogStd( 2 )( paramsPost ):annotate{ name = 'distThetaAccLapseSample' }
 	end

	d = nn.Narrow( 2, 1 , nsmpl-1                              )( samplePost ) 
	t = nn.Narrow( 2, 1 + nsmpl-1 , nsmpl-2                    )( samplePost ) 
	a = nn.Narrow( 2, 1 + nsmpl-1 + nsmpl-2 , (nsmpl-2)*dim    )( samplePost )

	if lapseDim > 1 then l = nn.Narrow( 2, 1 + nsmpl-1 + nsmpl-2 + (nsmpl-2)*dim, 1 )( samplePost ) end 

	--- run samples through transfer functions 

	if     distTransfer:find('Beta') then  

		local beta = distTransfer:gsub('Beta','')
		beta = tonumber( beta )
		d = nn.SoftPlus( beta )( d )

	elseif distTransfer:find('Sigmoid') then 

		local maxDist = distTransfer:gsub('Sigmoid', '')
		maxDist = tonumber( maxDist )
		d = nn.MulConstant( maxDist )( nn.Sigmoid()( d ) )

	elseif distTransfer == 'Exp' then 

		d = nn.Exp()( d ) 

	end
	d:annotate{ name = 'distSample' }

	a = nn.View( mb, nsmpl-2, dim )( a )
	a = nn.Transpose({2,3})( a ):annotate{ name = 'accSample' } 

	l = nn[lapseTransfer]()( l ) 
	l = nn.MulConstant( maxLapse - minLapse )( l ) 
	l = nn.AddConstant( minLapse )( l ):annotate{ name = 'lapse' }

	--- construct trajectory and evaluate likelihood 

	pABX = accCurvatureToZ( mb, dim, nsmpl )({ d, t, a }):annotate{ name = 'trajectory' }
	pABX = nn.View( mb * dim, nsmpl )( pABX )
	pABX = Differences( nsmpl )( pABX ):annotate{ name = 'differences' }
	pABX = nn.View( mb , dim, nsmpl, nsmpl )( pABX ) 
	pABX = nn.Square()( pABX ) 
	pABX = nn.Sum(2)( pABX ):annotate{ name = 'squared distances' } 
	pABX = nn.Sqrt()( pABX ):annotate{ name = 'distances' } 
	pABX = ABXpCorrect()( pABX ):annotate{ name = 'pABX' }


	local oneMinusLapse = nn.AddConstant( 1   )( nn.MulConstant( -1 )( l ) )
	oneMinusLapse = nn.Replicate( nsmpl^2, 2 )( oneMinusLapse ) 
	local  lapseOverTwo = nn.MulConstant( 0.5 )( l ) 
	lapseOverTwo  = nn.Replicate( nsmpl^2, 2 )( lapseOverTwo )
	pABX = nn.CAddTable()({nn.CMulTable()({pABX, oneMinusLapse}),lapseOverTwo}):annotate{ name = 'pCorrect' }


	local bcorrt = nn.AddConstant( 1e-8 )( pABX ) 
	bcorrt = nn.Log()( bcorrt ):annotate{ name = 'log p' } 
	bcorrt = wcorrt( bcorrt ):annotate{ name = 'weighted log p' }

	local bwrong = nn.MulConstant( -1 )( pABX ) 
	bwrong = nn.AddConstant( 1 + 1e-8 )( bwrong )
	bwrong = nn.Log()( bwrong ):annotate{ name = 'log 1-p' }
	bwrong = wwrong( bwrong ):annotate{ name = 'weighted log 1-p' } 

	local logLikelihood = nn.CAddTable()({ bcorrt, bwrong })
	logLikelihood = wcombi( logLikelihood ):annotate{ name = 'binomial' }
	logLikelihood = nn.View( mb, -1 )( logLikelihood )
	logLikelihood = nn.Sum( 2 )( logLikelihood )
	logLikelihood = nn.MulConstant( -1 )( logLikelihood )

	local loss 
	if #losses > 0 then

		table.insert( losses, logLikelihood )
		loss = nn.CAddTable()( losses ):annotate{ name = 'lossPerBatchSample' }

	else

		loss = logLikelihood

	end

	self.network = nn.gModule({ inode }, { loss })

	self:loadData( data )

end

function likelihood:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end

function likelihood:updateGradInput( input, gradOutput ) 

	self.gradInput = self.network:updateGradInput( input, gradOutput )

	return self.gradInput 

end

function likelihood:loadData( data )

	self.lgCorrt = self.lgCorrt or torch.Tensor( data[1]:size() ) 
	self.lgWrong = self.lgWrong or torch.Tensor( data[1]:size() ) 
	self.nAll    = self.nAll    or torch.Tensor( data[1]:size() ) 
	self.lgAll   = self.lgAll   or torch.Tensor( data[1]:size() ) 

	self.ncorrt:copy( data[1] ):add( 1 ) 
	self.nwrong:copy( data[2] ):add( 1 ) 
	self.nAll:copy( data[1] ):add( data[2] ):add( 1 ) 

	cephes.lgam( self.lgCorrt, self.ncorrt )
	cephes.lgam( self.lgWrong, self.nwrong )
	cephes.lgam( self.lgAll  , self.nAll ) 

	self.combin:copy( self.lgAll ):add( -1, self.lgCorrt ):add( -1, self.lgWrong ) 

	self.ncorrt:copy( data[1] )
	self.nwrong:copy( data[2] )

	self.mb = data[1]:size( 1 )
	self.nsmpl = data[1]:size( 2 )

end

