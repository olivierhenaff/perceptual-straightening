require 'nngraph'
require 'cephes'

local binomial, parent = torch.class('nn.BinomialNLL', 'nn.Module')

function binomial:__init( data )

	parent.__init(self)

	local wcorrt = nn.CMul( data[1]:size() ) 
	local wwrong = nn.CMul( data[2]:size() ) 
	local wcombi = nn.Add(  data[1]:size() ) 

	self.ncorrt = wcorrt.weight:zero() 
	self.nwrong = wwrong.weight:zero() 
	self.combin = wcombi.bias:zero()

	local pCorrect = nn.Identity()()

	local bcorrt = wcorrt( nn.Log()( pCorrect ):annotate{ name = 'log pCorrect' } )

	local bwrong = nn.AddConstant( 1+1e-8 )( nn.MulConstant(-1)( pCorrect ) )
	bwrong = wwrong( nn.Log()( bwrong ):annotate{ name = 'log 1 - pCorrect' } ) 

	local nll = wcombi( nn.CAddTable()({ bcorrt, bwrong }) )

	-- nll = nn.MulConstant(-1)( nn.Sum(2)( nn.View( 1, -1 )( nll ) ) ) 
	nll = nn.MulConstant(-1)( nn.Sum(2)(nll) ) 

	self.network = nn.gModule({ pCorrect }, { nll })

	self:loadData( data )

end

function binomial:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end 

function binomial:updateGradInput( input, gradOutput )

	self.gradInput = self.network:updateGradInput( input, gradOutput ) 

	return self.gradInput

end 

function binomial:loadData( data )

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

end

--[[
local binomial, parent = torch.class('nn.Binomial', 'nn.Sequential')

function binomial:__init( data )

	parent.__init(self)

	local wcorrt = nn.CMul( data[1]:size() ) 
	local wwrong = nn.CMul( data[2]:size() ) 
	local wcombi = nn.Add(  data[1]:size() ) 

	self.ncorrt = wcorrt.weight:zero() 
	self.nwrong = wwrong.weight:zero() 
	self.combin = wcombi.bias:zero()

	local seq1 = nn.Sequential() 
	seq1:add( nn.Log() ) 
	seq1:add( wcorrt )

	local seq2 = nn.Sequential() 
	seq2:add( nn.MulConstant( -1 ) ) 
	seq2:add( nn.AddConstant(  1 + 1e-8 ) ) 
	seq2:add( nn.Log() ) 
	seq2:add( wwrong )

	local cat = nn.ConcatTable()
	cat:add( seq1 ) 
	cat:add( seq2 ) 

	self:add( cat )
	self:add( nn.CAddTable() )
	self:add( wcombi )

	self:loadData( data )

end
]]
