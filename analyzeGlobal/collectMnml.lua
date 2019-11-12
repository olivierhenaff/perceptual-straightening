function bootstrappedStats( bootstrapped, exp )

	if #bootstrapped > 0 then 

		for i = #bootstrapped, 1, -1 do 
			local m = bootstrapped[i]
			if m ~= m then table.remove( bootstrapped, i ) end 
		end 

		bootstrapped = torch.Tensor( bootstrapped )

		if bootstrapped:size(1) > 1 then 

			exp.mean = bootstrapped:mean()
			exp.std  = bootstrapped:std()

			local sorted, sortInds = torch.sort( bootstrapped, 1 )

			local ciVal  = {} 
			local ciPerc = { 0.025, 0.16, 0.50, 0.84, 0.975 }

			for _, p in pairs( ciPerc ) do 

				ciVal[100*p] = sorted[ math.ceil( p * bootstrapped:size(1) ) ]

			end

			exp.quantiles = ciVal

			exp.bootstrapped = bootstrapped

		end

	end

end	

function solutionNetwork( solution, data, sampleMode ) 

	if not data.all then 

		local nsmpl = 11 
		local dummy = torch.rand(nsmpl,nsmpl)
		data = {right = dummy:clone(), wrong = dummy:clone():mul(-1):add(1), all = dummy:clone():fill(1), pCorrect = dummy:clone() }

	elseif data.all:dim() > 2 then 

		data = {right = data.right[s], wrong = data.wrong[s], all = data.all[s], pCorrect = data.pCorrect[s]}

	end

	local polarInit = initializeDistThetaAccBootstrap( data, 1 )
	polarInit = initializeDistThetaAccPrior( polarInit, 3 )

	if lapseDim then polarInit = initializeLapse( polarInit, lapseDim ) end 

	local d = polarInit[1] 
	local t = polarInit[2]
	local a = polarInit[3] 
	local nZ = d:size( 2 ) + t:size( 2 ) + a:size(2)*a:size(3) + lapseDim - 1
	local distThetaAccRotation = torch.Tensor( nZ, nZ ):zero()
	table.insert( polarInit, distThetaAccRotation )

	polarInit = initializeDistThetaAccBatch( polarInit, mb )

	local network = nn.ABX_NLL_MNML_polar_VB( {data.right, data.wrong}, dim, mb, polarInit, sampleMode )
	network:updateOutput( solution )

	return network

end

function loadSolutionNetwork( domain, sequence, condition, subject )

	local loaddir = makeDirGlobal( domain, sequence, condition, subject ) 
	local lossdir = loaddir ..    'losses/'
	local soludir = loaddir ..  'solution/'

	local file = 'run1.t7'
	local losses     = torch.load( lossdir .. file )
	local finalLoss  = losses:narrow( 1, collectLossesFrom+1, maxiter-collectLossesFrom ):mean(1):squeeze()
	local _, sortInd = torch.sort( finalLoss )
	local bestInd    = sortInd[1]

	local solution = torch.load( soludir .. file )

	network = network or solutionNetwork( solution, data ) 
	network:updateOutput( solution )

	return solution, network, bestInd

end

function collectMnml( domain ) 

	print( 'collecting' )

	local network 
	local experimentList = {} 

	local savedir = makeDirGlobal( domain ) 
	os.execute( 'mkdir -p ' .. savedir )

	local mb    = 1 
	local nsmpl = 11 --trajectoryCntrl:size(1)
	local dim   = 10 --trajectoryCntrl:size(2)
	local moog  = accCurvatureToZ( mb, dim, nsmpl )

	local distTransferNet 
	if distTransfer:find('Sigmoid') then 
		local maxDist = distTransfer:gsub('Sigmoid', '')
		maxDist = tonumber( maxDist )
		distTransferNet = nn.Sequential():add( nn.Sigmoid() ):add( nn.MulConstant( maxDist ) )
	elseif distTransfer:find('Beta') then
		local beta = distTransfer:gsub('Beta', '')
		beta = tonumber( beta )
		distTransferNet = nn.SoftPlus( beta ) 
	elseif distTransfer == 'Exp' then 
		distTransferNet = nn.Exp() 
	end


	for s, sequence in pairs( sequences ) do

		for _, condition in pairs( conditions ) do 

			local x = loadSequence( condition, sequence ) 
			local cPointwise = computeCurvaturePointwise( x ):div( math.pi ) 
			local cPixel = cPointwise:mean()
			print('cPixel', cPixel )

			if not ( condition:find( 'contrast' ) and s > 1 ) then 

				for _, subject in pairs( subjects ) do 

					local data = loadDataRec( condition, subject, tags, max_deviation, nil, natOnly )

					if data.all or condition:find('curvature') then 

						local loaddir = makeDirGlobal( domain, sequence, condition, subject ) 
						local lossdir = loaddir ..    'losses/'
						local soludir = loaddir ..  'solution/'

						local bootTheta  = {} 
						local bootDist   = {}
						local  expTheta  = {} 
						local  expDist   = {} 
						local detail     = {}

						local trajectory 

						local count = 0 
						local time = sys.clock() 
						local time_prev = time 

						for file in paths.files( soludir ) do

							if file:find( 'run' ) then 

								local runInd = file:gsub('run',''):gsub('.t7',''); runInd = tonumber( runInd )

								if paths.filep( lossdir .. 'run' .. runInd .. '.t7' ) then 

									collectgarbage() 

									count = count + 1 

									local losses  = torch.load( lossdir .. 'run' .. runInd .. '.t7' )
									local finalLoss = losses:narrow( 1, collectLossesFrom+1, maxiter-collectLossesFrom ):mean(1):squeeze()
									local _, sortInd = torch.sort( finalLoss )
									local bestInd = sortInd[1]

									local solution = torch.load( soludir .. file )

									network = network or solutionNetwork( solution, data, { all = true } ) 
									network:updateOutput( solution )

									local priorPreTheta = findNode( network.network, 'priorTheta' ).output[bestInd]:squeeze()
									local globalTheta   = priorPreTheta:narrow(1,1,1):squeeze() / math.pi 

									local priorPreDist = findNode( network.network, 'priorDist'  ).output[bestInd]:squeeze()
									local globalDist   = distTransferNet:updateOutput( priorPreDist:narrow(1,1,1) ):squeeze()

									table.insert( bootTheta , globalTheta )
									table.insert( bootDist  , globalDist  ) 

									if runInd == 1 then 

										trajectory      = findNode( network.network, 'trajectory' ).output[bestInd]:squeeze():clone()
										local preDists  = findNode( network.network, 'preDistParams' ).output[bestInd]:squeeze():clone()
										detail.dists    = distTransferNet:updateOutput( preDists[1]                  ):squeeze():clone()
										detail.thetas   = findNode( network.network, 'thetaParams01' ).output[bestInd]:squeeze():clone()
										detail.lapse    = findNode( network.network, 'lapse'         ).output:squeeze()[bestInd]
										detail.pCorrect = findNode( network.network, 'pCorrect'      ).output[bestInd]:squeeze():clone()

										expTheta.map  = globalTheta
										 expDist.map  = globalDist

									end

								end

							end

						end

						bootstrappedStats( bootTheta , expTheta ) 
						bootstrappedStats( bootDist  , expDist  ) 

						local experiment = { dist = expDist, theta = expTheta, 
												sequence = sequence, condition = condition, subject = subject,
												pixelTheta = cPixel, trajectory = trajectory, detail = detail, count = count,
												 }

						torch.save( savedir .. sequence .. '_' .. condition .. '_' .. subject .. '.t7', experiment ) 

						print( sequence, condition, subject, count, 'counts')

						collectgarbage()

					end
				end
			end
		end
	end

end