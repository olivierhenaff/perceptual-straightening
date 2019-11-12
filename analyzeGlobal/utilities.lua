require 'optim'
require 'analyzeGlobal/utilitiesPolar'

function round(num, idp)
	local mult = 10^(idp or 0)
	return math.floor(num * mult + 0.5) / mult
end

function loadExpList( domain )

	local expList = {}
	local loaddir = makeDirGlobal(domain)

	for _, sequence in pairs( sequences ) do

		for _, condition in pairs( conditions ) do 

			for _, subject in pairs( subjects ) do

				local loaddir = loaddir .. sequence .. '_' .. condition .. '_' .. subject .. '.t7'

				if paths.filep( loaddir ) then

					table.insert( expList, torch.load( loaddir ) )

				end

			end

		end

	end

	return expList

end

function tableToSet(t)

	local s = {} 
	for _, x in pairs(t) do
		s[x] = true 
	end
	return s 
end

function capFirst( str ) 

	return str:sub(1,1):upper()..str:sub(2,-1)

end

function makeDirGlobal( domain, sequence, condition, subject )

	local model = model or 'global'

	local savedir = '/scratch/ojh221/results/untangling/' .. model .. '/'

	if domain then savedir = savedir .. domain    .. '/' end 

	savedir = savedir .. 'maxDeviation' .. max_deviation

	if domain == 'synthetic'  and dataLapse then savedir = savedir .. '_dataLapse' .. dataLapse           end 
	if domain == 'pixel'      and dataAcc   then savedir = savedir .. '_dataAcc'   .. capFirst( dataAcc ) end 
	if dataMultiplier                       then savedir = savedir .. '_dataMult'  .. dataMultiplier      end 
	if domain == 'synthetic'  and dataDim   then savedir = savedir .. '_dataDim'   .. dataDim             end 

	if dithered then 
		if domain == 'synthetic' then 
			savedir = savedir .. '_nTrials'  .. nTrials
		else
			savedir = savedir .. '_dithered' .. dithered
		end
	end
	savedir = savedir .. '/' 

	if sequence  then savedir = savedir .. sequence  .. '/' end 
	if condition then savedir = savedir .. condition .. '/' end
	if subject   then savedir = savedir .. subject   .. '/' end 

	if summary then savedir = savedir .. 'summary' .. summary:sub(1,1):upper()..summary:sub(2,-1) end 

	if dim then savedir = savedir .. 'dim'           .. dim end 

	if mb then savedir = savedir .. '_mb' .. mb end
	if lr then savedir = savedir .. '_lr' .. lr end 
	if maxiter then savedir = savedir .. '_maxiter' .. maxiter end 

	if lrDrop then
		for iter, drop in pairs(lrDrop) do 
			savedir = savedir .. '_drop' .. iter
		end 
	end

	if collectLossesFrom then savedir = savedir .. '_collect' .. collectLossesFrom end 

	if expAveraging then savedir = savedir .. '_expAveraging' .. expAveraging end 


	if rotateFirst  then savedir = savedir .. '_rotateFirst' end 

	if sampleMode then 

		savedir = savedir .. '_sampleMode' 

		if sampleMode.d   then savedir = savedir .. 'D'   end 
		if sampleMode.t   then savedir = savedir .. 'T'   end 
		if sampleMode.a   then savedir = savedir .. 'A'   end 
		if sampleMode.l   then savedir = savedir .. 'L'   end 
		if sampleMode.all then savedir = savedir .. 'ALL' end 

	end

	if lapseDim      then savedir = savedir .. '_lapseDim' .. lapseDim   end 
	if lapseTransfer then savedir = savedir .. '_lapse' .. lapseTransfer end 
	if maxLapse      then savedir = savedir .. '_maxLapse' .. maxLapse   end 
	if minLapse      then savedir = savedir .. '_minLapse' .. minLapse   end 

	-- if thetaTransfer then savedir = savedir .. '_thetaTransfer' .. thetaTransfer end
	if  distTransfer then savedir = savedir ..  '_distTransfer' ..  distTransfer end 

	if accPrior      then savedir = savedir .. '_accPrior' .. capFirst( accPrior ) end 

	if zPost         then savedir = savedir .. '_zPost'    .. capFirst(   zPost  ) end 

	if nbootstrap    then savedir = savedir .. '_initBoot' .. nbootstrap end 

	if distInit      then savedir = savedir .. '_initDist' .. distInit end 

	if bootMeth then savedir = savedir .. '_boot' .. bootMeth:sub(1,1):upper()..bootMeth:sub(2,-1) end
	if repeatNumber then savedir = savedir .. '_repeat' .. repeatNumber end 

	savedir = savedir .. '/'

	return savedir 

end

function listSequencesConditionsSubjects( sequences, conditions, subjects, saveImages )

	local list = {} 

	for s, sequence in pairs( sequences ) do 

		for _, condition in pairs( conditions ) do

			for _, subject in pairs( subjects ) do

				local data = loadDataRec( condition, subject, tags, max_deviation, nil, natOnly )

				if data.all then 

					table.insert( list, {sequence = sequence, condition = condition, subject = subject} )

				end 

			end

		end

	end

	return list 

end


function initializeTrajectory( pCorrect, dPrimeInit, fullDim )

	local dPrime 

	if dPrimeInit then 

		dPrime = dPrimeInit

	else

		local dPrimeMax  = distInit
		local nPointsInv = 10000
		local pABXinv    = ABXpCorrectInverse( dPrimeMax, nPointsInv )
		dPrime           = pABXinv:updateOutput( pCorrect )

	end

	-- print( dPrime ) 

	local g  = dPrime:clone():pow(2)
	local m1 = g:mean(1)
	local m2 = g:mean(2)
	local m  = g:mean() 

	g:add( -1, m1:expandAs( g ) ) 
	g:add( -1, m2:expandAs( g ) )
	g:add(  m ) 
	g:div( -2 ) 

	local u, s, v = torch.svd( g ) 

	if not fullDim then 

		u = u:narrow( 2, 1, dim )
		s = s:narrow( 1, 1, dim ) 
		v = v:narrow( 2, 1, dim ) 

	end

	local init = u * torch.diag( s:clone():pow(0.5) ) --[nsmpl][dim]

	return init 

end