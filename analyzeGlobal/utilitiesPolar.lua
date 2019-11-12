-- require 'common-torch/utilities'
require 'analyzeGlobal/utilitiesPolarVB'
require 'modules/abx'

function softPlusInv( y )

	local x = y:clone():exp():add(-1):log() 

	return x 

end 

function sigmoidInv( y )

	local x = y:clone():pow(-1):add(-1):log():mul(-1)

	return x 

end 

function testSoftPlusInv()

	local s = nn.SoftPlus()
	local x = torch.linspace(-3,3,101)
	local y = s:updateOutput( x ) 
	local z = softPlusInv( y ) 

	print( 'soft plus inv err', x:dist( z ) ) 

end 

function testSigmoidInv()

	local s = nn.Sigmoid()
	local x = torch.linspace(-3,3,101)
	local y = s:updateOutput( x ) 
	local z = sigmoidInv( y ) 

	print( 'sigmoid inv err', x:dist( z ) ) 

end 

function grahamSchmidtInplace( x ) 

	local nsmpl = x:size(1) 

	for i = 1, nsmpl do 

		for j = 1, i-1 do x[i]:add( - x[j]:dot( x[i] ), x[j] ) end 

		x[i]:div( x[i]:norm() )

	end 

end

function alignTrajectoryCanonical( trajInit ) 

	local nsmpl = trajInit:size( 1 ) 
	local   dim = trajInit:size( 2 ) 

	local z0 = trajInit[1]:clone()
	for i = 1, nsmpl do trajInit[i]:add( -1, z0 ) end 

	local u = torch.randn( dim, dim ) 
	u[1]:copy( trajInit[2] ) 
	grahamSchmidtInplace( u ) 

	trajInit = trajInit * u:t() 

	return trajInit 

end

function initializeDistThetaAcc( z ) -- z:[nsmpl][dim] 

	local nsmpl = z:size(1)
	local   dim = z:size(2)

	local d = torch.Tensor( nsmpl-1 )
	local t = torch.Tensor( nsmpl-2 ) 
	local a = torch.Tensor( nsmpl-2, dim ) 

	local v    = torch.Tensor( nsmpl-1, dim ) 

	for i = 1, nsmpl-1 do 

		v[i]:copy( z[i+1] ):add( -1, z[i] )
		d[i] = v[i]:norm() 
		v[i]:div( d[i] ) 

	end 

	for i = 1, nsmpl-2 do 

		local cosTheta = v[i+1]:dot( v[i] )
		cosTheta = math.min( cosTheta,  1 )
		cosTheta = math.max( cosTheta, -1 ) 
		t[i] = math.acos( cosTheta )

		a[i]:copy( v[i+1] ):add( -cosTheta, v[i] )
		a[i]:div( a[i]:norm() ) 

	end

	if     distTransfer:find( 'Beta' ) then

		local beta = distTransfer:gsub( 'Beta', '' ) 
		beta = tonumber( beta ) 
		d:mul( beta ):exp():add( -1 ):log():div( beta ) 

	elseif distTransfer:find( 'Sigmoid' ) then 

		local maxDist = distTransfer:gsub('Sigmoid', '')
		maxDist = tonumber( maxDist )
		d = sigmoidInv( d:clone():div( maxDist ) )

	elseif distTransfer == 'Exp' then 

		d:log()

	else

		error('unknown transfer function for distance')

	end

	if     thetaTransfer == 'Sigmoid' or thetaTransfer == 'SigmoidIdentity' then 

		t =  sigmoidInv( t:clone():div(math.pi) )

	elseif thetaTransfer == 'Identity' then 
	else

		error('unknown transfer function for theta')

	end

	local init = {d, t, a }

	return init 

end

function inferLatentsPolar( p )

	local trajInit  = initializeTrajectory( p )
	trajInit        = alignTrajectoryCanonical( trajInit )
	local polarInit = initializeDistThetaAcc(   trajInit )

	return polarInit

end 

function initializeDistThetaAccBootstrap( data ) 

	local polarInit             = inferLatentsPolar( data.pCorrect ) 
	local finalInit 			= {} 

	for i = 1, #polarInit do 

		finalInit[i] = appendBatchDimTensor( polarInit[i], 2 )
		finalInit[i][1]:copy( polarInit[i] ) 
		finalInit[i][2]:fill( -3 )

	end

	return finalInit

end 

function appendBatchDimTensor( tensor, nfeat )

	local size = torch.LongStorage( tensor:dim()+1 )
	size[1] = nfeat
	for j = 1, tensor:dim() do size[j+1] = tensor:size(j) end

	return torch.Tensor( size )

end 

function initializeLapse( init, lapseDim )

	local lapse = torch.Tensor(lapseDim):fill(0)
	lapse[1] = -10

	table.insert( init, lapse )

	return init

end

function initializeDistThetaAccBatch( init, mb )

	local appended = {} 

	for i = 1, #init do

		appended[i] = appendBatchDimTensor( init[i], mb )

		for j = 1, mb do appended[i][j]:copy( init[i] ) end

	end 
	init = appended

	local nparams = 0; for i = 1, #init do nparams = nparams + init[i]:nElement()/mb end 
	local fused = torch.Tensor( mb, nparams ) 
	local segm = Segment( init, 2 ); segm:updateOutput( fused ) 
	z = segm:updateGradInput( fused, init ) 

	return init, z 

end 