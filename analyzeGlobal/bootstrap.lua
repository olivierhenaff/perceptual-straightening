require 'nngraph'

nngraph.setDebug(true)

require 'stimuli/load'
require 'utilities/processData'
require 'utilities/evaluateCurvature'
require 'utilities/synthesizeData'
require 'analyzeGlobal/utilities'
require 'analyzeGlobal/gnuplot'
require 'analyzeGlobal/collectMnml'
require 'modules/abxMoog'

require 'analyzeGlobal/utilities'


cmd = torch.CmdLine()
cmd:option('-seed',  1, 'Use positive seeds to analyze the data. Seed=1 will analyze the first subject/sequence pair, seed=2 the second, etc. Seeds greater than the total number of subject/sequence pairs will analyze bootstrapped versions of the same data. In paper we use 100 bootstrapped samples for every sequence/subject pair.')
-- cmd:option('-seed', -1, 'Use seed = -1 to collect the results of this analysis. In particular, this will compute summary statistics across bootstrapped samples, so that you don\'t need to load all of them after.')
-- cmd:option('-seed',  0, 'Use seed =  0 to display the collected results. Useful for plotting, examining different statistics, etc. This part is very fast, since it only loads the summary statistics across bootstrapped samples (for every subject/sequence pair).')

cmd:option('-domain', 'perceptual', 'pixel or perceptual') -- use 'perceptual' to analyze subjects' data. 
-- cmd:option('-domain', 'pixel'     , 'pixel or perceptual') -- use 'pixel' to construct 'control' trajectories based off of the inferred perceptual trajectories (in which the curvature has been replaced by pixel domain curvatures), simulate perceptual datasets from these trajectories, and analyze them. Should give curvature estimates that are close to pixel-domain curvature. 

cmd:option('-dim'         , 10, 'dimensionality of observer model')
cmd:option('-repeatNumber',  1,'index of this fitting + recovery procedure') -- allows you to repeat the whole procedure a few times, in case you were wondering how reliable the whole thing was. 

params = cmd:parse(arg)

-- model = 'data_yoon'
model = 'data_corey'

bootMeth = 'nonparam'

dim             = params.dim
parametrization = 'polar'
inference       = 'VB'

mb = 6 
lr = 0.01
expAveraging = 0.999 

maxiter           = 80000
collectLossesFrom = 50000 

thetaTransfer = 'SigmoidIdentity'
distTransfer  = 'Sigmoid10'

 distPrior = 'scalar'
thetaPrior = 'scalar'
  accPrior = 'zeroMean'
zPost      = 'full'

distInit = 5

repeatNumber = params.repeatNumber


natOnly       = true
max_deviation = 2
dataLapse     = 0.12 

-- dataAcc = 'orig'
dataAcc = 'origReally'

 minLapse     = 0
 maxLapse     = 0.12
lapseDim      = 2
lapseTransfer = 'cdfNormal'

domain = params.domain 

tags       = {'big'}


if model == 'data_yoon' then 

	-- sequences  = {'groundtruth'}
	-- subjects   = {'qj', 'yb'} 
	-- conditions = {'natural_fovea_04_EGOMOTION', 'natural_fovea_05_PRAIRIE', 'natural_parafovea_04_EGOMOTION', 'natural_parafovea_05_PRAIRIE', 'natural_parafovea_06_DAM', 'natural_periphery_05_PRAIRIE', 'natural_periphery_06_DAM', 'synthetic_fovea_04_EGOMOTION', 'synthetic_fovea_05_PRAIRIE', 'synthetic_parafovea_04_EGOMOTION', 'synthetic_parafovea_06_DAM', 'synthetic_periphery_05_PRAIRIE', 'synthetic_periphery_06_DAM'}

	sequences  = {'groundtruth'}
	subjects   = {'alexandra', 'carlos', 'maddy', 'ryan'}
	conditions = {'natural_parafovea_04_EGOMOTION', 'natural_periphery_06_DAM', 'synthetic_fovea_06_DAM', 'synthetic_parafovea_04_EGOMOTION', 'natural_parafovea_05_PRAIRIE', 'synthetic_fovea_04_EGOMOTION', 'natural_fovea_05_PRAIRIE', 'natural_parafovea_04_EGOMOTION', 'synthetic_fovea_06_DAM', 'synthetic_parafovea_05_PRAIRIE', 'natural_fovea_05_PRAIRIE', 'natural_parafovea_04_EGOMOTION', 'natural_periphery_05_PRAIRIE', 'natural_periphery_06_DAM', 'synthetic_fovea_06_DAM', 'synthetic_parafovea_04_EGOMOTION', 'synthetic_periphery_05_PRAIRIE'}

elseif model == 'data_corey' then 

	sequences  = {'groundtruth'}
	-- subjects   = {'cmz'}
	-- conditions = {'pilot_movie1', 'pilot_movie2'}
	-- subjects   = {'CMZ'}
	subjects   = {'cmz', 'CMZ'}
	-- conditions = {'PredControl_carnegie-dam', 'PredControl_leaves-wind'}
	-- conditions = {'PredControl_ice3Mod', 'PredControl_beesMod', 'PredControl_water', 'PredControl_butterflies0'}
	-- conditions = {'PredControl_prairie1Con', 'PredControl_chironomusMod'}
	conditions = {'pilot_movie1', 'pilot_movie2', 'PredControl_carnegie-dam', 'PredControl_leaves-wind', 'PredControl_ice3Mod', 'PredControl_beesMod', 'PredControl_water', 'PredControl_butterflies0', 'PredControl_prairie1Con', 'PredControl_chironomusMod'}

end

if params.seed > 0 then

	local list = listSequencesConditionsSubjects( sequences, conditions, subjects, params.seed == 1 )

	-- print( list )
	-- error('stop here')

	if model == 'data_yoon' then 
		list = {
		{sequence = 'groundtruth', subject = 'alexandra', condition = 'natural_parafovea_04_EGOMOTION'}, 
		{sequence = 'groundtruth', subject = 'alexandra', condition = 'natural_periphery_06_DAM'}, 
		{sequence = 'groundtruth', subject = 'alexandra', condition = 'synthetic_fovea_06_DAM'}, 
		{sequence = 'groundtruth', subject = 'alexandra', condition = 'synthetic_parafovea_04_EGOMOTION'}, 
		{sequence = 'groundtruth', subject = 'carlos', condition = 'natural_parafovea_05_PRAIRIE'}, 
		{sequence = 'groundtruth', subject = 'carlos', condition = 'synthetic_fovea_04_EGOMOTION'}, 
		{sequence = 'groundtruth', subject = 'maddy', condition = 'natural_fovea_05_PRAIRIE'}, 
		{sequence = 'groundtruth', subject = 'maddy', condition = 'natural_parafovea_04_EGOMOTION'}, 
		{sequence = 'groundtruth', subject = 'maddy', condition = 'synthetic_fovea_06_DAM'}, 
		{sequence = 'groundtruth', subject = 'maddy', condition = 'synthetic_parafovea_05_PRAIRIE'}, 
		{sequence = 'groundtruth', subject = 'ryan', condition = 'natural_fovea_05_PRAIRIE'}, 
		{sequence = 'groundtruth', subject = 'ryan', condition = 'natural_parafovea_04_EGOMOTION'}, 
		{sequence = 'groundtruth', subject = 'ryan', condition = 'natural_periphery_05_PRAIRIE'}, 
		{sequence = 'groundtruth', subject = 'ryan', condition = 'natural_periphery_06_DAM'}, 
		{sequence = 'groundtruth', subject = 'ryan', condition = 'synthetic_fovea_06_DAM'}, 
		{sequence = 'groundtruth', subject = 'ryan', condition = 'synthetic_parafovea_04_EGOMOTION'}, 
		{sequence = 'groundtruth', subject = 'ryan', condition = 'synthetic_periphery_05_PRAIRIE'}, 
		}
	end

	local expInd = ( params.seed - 1 ) % #list + 1 
	local runInd = math.ceil( params.seed / #list )

	local experiment = list[ expInd ] 

	conditions = { experiment.condition }
	subjects   = { experiment.subject   }
	sequences  = { experiment.sequence  }

	if domain:find('pixel') then 

		exp = torch.load( makeDirGlobal('perceptual') .. experiment.sequence .. '_' .. experiment.condition .. '_' .. experiment.subject .. '.t7' )

	end

	params.seed = runInd 

end

sequence = sequences[1]
s = sequence == 'groundtruth' and 1 or sequence == 'pixelfade' and 2 

function bootstrap()

	for _, cond in pairs( conditions ) do

		condition = cond

		if not ( condition:find( 'contrast' ) and s > 1 ) then

			for _, subj in pairs( subjects ) do

				subject = subj

				print( condition, subject, sequence )

				local data = loadDataRec( condition, subject, tags, max_deviation, nil, natOnly, dithered )

				if data.all or condition:find('curvature') then 

					local savedir = makeDirGlobal( domain, sequence, condition, subject ) 
					local bootdir = savedir .. 'bootstrap/'
					local lossdir = savedir ..    'losses/'
					local soludir = savedir ..  'solution/'

					os.execute('mkdir -p ' .. bootdir )
					os.execute('mkdir -p ' .. lossdir )
					os.execute('mkdir -p ' .. soludir )

					if domain == 'pixel' then 

						local datadir = savedir .. 'data.t7'

						if paths.filep( datadir ) then 

							data = torch.load( datadir )

						else 

							local x          = loadSequence( exp.condition, exp.sequence ) 
							local cPointwise = computeCurvaturePointwise( x ):div( math.pi ) 
							local mb    =  1
							local nsmpl = 11 
							local theta = torch.Tensor( mb,      nsmpl-2 ):copy( cPointwise )

							local traj    = exp.trajectory:t()
							local d, c, a = computeDistCurvAcc( traj )
							local acc = torch.Tensor( mb, dim, nsmpl-2 ):copy( a ) 

							local dist = torch.Tensor( mb, nsmpl-1 ):copy( exp.detail.dists  )

							local moog     = abxMoog(      mb, dim, nsmpl )
							local pCorrect = moog:updateOutput({ dist, theta, acc }):squeeze() 

							local lapse = exp.detail.lapse 
							pCorrect:mul( 1 - lapse ):add( lapse / 2 ) 

							data = synthesizePerceptualData( pCorrect, data.all[s] )

							torch.save( datadir, data ) 

						end

					elseif domain == 'perceptual' then

						data.right    = data.right[s]
						data.wrong    = data.wrong[s]
						data.all      = data.all[s] 
						data.pCorrect = data.pCorrect[s] 

					end

					local theta, losses, z = inferGlobalCurvaturePolarVB( data, maxiter, gradCrit, dim )

					local runInd = params.seed
					local hopInt = 50
					while paths.filep( soludir .. 'run' .. runInd .. '.t7' ) do runInd = runInd + hopInt end

					print( 'saving with runInd', runInd )

					if theta  then torch.save( bootdir .. 'run' .. runInd .. '.t7', theta  ) end 
					if losses then torch.save( lossdir .. 'run' .. runInd .. '.t7', losses ) end 
					if z      then torch.save( soludir .. 'run' .. runInd .. '.t7', z      ) end 

				end

			end
		end
	end
end


if     params.seed == -1 then 

	collectMnml(  domain ) 

elseif params.seed == 0  then 

	local exp = loadExpList( 'perceptual' )

	local report = {}
	for i = 1, #exp do 
		local e    = exp[i]
		local name = e.subject .. '_' .. e.condition 
		-- print(e.theta)
		-- report[name] = round(180*e.theta.quantiles[16],1)  .. ',' .. round(180*e.theta.mean,1) .. ',' .. round(180*e.theta.quantiles[84],1) .. ',' .. 180*e.pixelTheta
		report[name] = round(180*e.theta.quantiles[2.5],1)  .. ',' .. round(180*e.theta.mean,1) .. ',' .. round(180*e.theta.quantiles[97.5],1) .. ',' .. 180*e.pixelTheta
		-- report[name] = round(180*e.theta.mean,1) .. ',' .. round(180*e.theta.std,1) .. ',' .. 180*e.pixelTheta
		-- report[name] = round(180*e.theta.mean,1) .. ',' .. 180*e.pixelTheta
		-- report[name] = round(180*e.theta.mean,2) .. ',' .. 180*e.pixelTheta
		-- local x = loadSequence( e.condition, e.sequence ) 
		-- local cPointwise = computeCurvaturePointwise( x ):div( math.pi ) 
		-- local cPixel = cPointwise:mean()
		-- report[name] = report[name] .. ',' .. 180*cPixel
	end

	for u,v in pairs( report ) do print( u .. ', ' .. v ) end

else

	bootstrap()

end