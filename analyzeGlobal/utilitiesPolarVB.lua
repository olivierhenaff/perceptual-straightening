require 'modules/abxGlobalNLLcurvature-mnml-VB'
require 'analyzeGlobal/utilitiesRotation'

function findNode( gmodule, nodeName )
	for u,v in pairs( gmodule.forwardnodes ) do 
		local name = v.data.annotations.name
		if name and name == nodeName then 
			return v.data.module
		end
	end
end

function initializeThetaNoisePrior( init )

	table.insert( init, torch.Tensor(2):fill(0) )

	return init

end

function initializeDistThetaAccPrior( polarInit, maxPrior )

	local maxPrior = maxPrior or #polarInit 

	for i = 1, math.min( #polarInit, maxPrior ) do 

		local priorInit = torch.Tensor( polarInit[i]:select( 2, 1 ):size() )

		local mu  = polarInit[i][1]:mean( 1 )
		priorInit:narrow(1,1,1):copy( mu  ) 

		local var = polarInit[i][1]:var( 1 )
		priorInit:narrow(1,2,1):copy( var )
		priorInit:narrow(1,2,1):add( polarInit[i][2]:clone():mul(2):exp():mean(1) )
		priorInit:narrow(1,2,1):log():div(2)

		table.insert( polarInit, priorInit )

	end 

	return polarInit 

end

function initializePolarAll( data ) 

	local polarInit, z 
	polarInit = initializeDistThetaAccBootstrap( data )
	polarInit = initializeDistThetaAccPrior( polarInit, 3 )

	if lapseDim then polarInit = initializeLapse( polarInit, lapseDim ) end 

	local d = polarInit[1] 
	local t = polarInit[2]
	local a = polarInit[3] 
	local nZ = d:size( 2 ) + t:size( 2 ) + a:size(2)*a:size(3) + lapseDim - 1 
	local distThetaAccRotation = torch.Tensor( nZ, nZ ):zero()
	table.insert( polarInit, distThetaAccRotation )

	polarInit, z = initializeDistThetaAccBatch( polarInit, mb )

	return polarInit, z 

end

function inferGlobalCurvaturePolarVB( data, maxiter, gradCrit, dim )

	local configAdam = { learningRate = lr }
	local state      = {} 

	local nsmpl = data.all:size( data.all:dim() ) 

	if params.seed > 1 and bootMeth == 'nonparam' then 

		randomkit.binomial( data.right, data.all:clone(), data.pCorrect )
		data.wrong:copy( data.all ):add( -1, data.right )

	end

	local gradOutput = torch.Tensor( mb ):fill( 1 ) 

	local polarInit, z = initializePolarAll( data )

	local zAverage
	if expAveraging then zAverage = z:clone():zero() end

	local network = nn.ABX_NLL_MNML_polar_VB( {data.right, data.wrong}, dim, mb, polarInit, sampleMode )

	local function opfunc( z )

		local loss = network:updateOutput( z )
		local gz   = network:updateGradInput( z, gradOutput )

		local m = loss:mean()
		if m ~= m then 

			local o = z 
			print( 'z', o:min(), o:mean(), o:max() ) 
			print( o:size() )

			local nodes = {'trajectory', 'differences', 'distances', 'pABX', 'log p', 'log 1-p', 'weighted log p', 'weighted log 1-p', 'binomial' }
			for _, node in pairs( nodes ) do 

				local o = findNode( network.network, node ).output
				print( node, o:min(), o:mean(), o:max() ) 
				print( o:size() )

			end

			print('a nan has appeared in loss! loss = ' )
			print( loss ) 
			error('a nan has appeared in loss!')

		end

		return loss[1], gz

	end

	local losses = torch.Tensor( maxiter, mb )
	local thetas = torch.Tensor( maxiter, mb )
	local networkThetas
	for t = 1, maxiter do

		if configAdam and lrDrop and lrDrop[t] then configAdam.learningRate = configAdam.learningRate/lrDrop[t] end 

		if t < collectLossesFrom then 			

			optim.adam( opfunc, z, configAdam, state )

			if expAveraging then zAverage:mul( expAveraging ):add( 1 - expAveraging, z ) end

		else

			if expAveraging and t == collectLossesFrom then z:copy( zAverage ):div( 1 - expAveraging^collectLossesFrom ) end

			network:updateOutput( z )

		end

		if t % 100 == 0 then 

			print( 'iteration', t, 'minimum loss', network.output:min() )

		end

		if t == 1 then networkThetas = findNode( network.network, 'thetaParams01' ).output end

		losses[t]:copy( network.output )
		thetas[t]:copy( networkThetas:select( 2, 1 ):mean(2) )

	end

	return thetas, losses, z

end




