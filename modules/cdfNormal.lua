require 'nn'
require 'cephes'

require 'modules/GaussianLikelihood'

local cdf, parent = torch.class('nn.cdfNormal', 'nn.Module')

function cdf:updateOutput( input ) 

	self.output:resizeAs( input ) 

	cephes.ndtr( self.output, input )

	return self.output

end 

function cdf:updateGradInput( input, gradOutput ) 

	self.gradInput:resizeAs( input ):copy( input )

	self.gradInput:pow( 2 ):div( - 2 ):exp():div( math.sqrt( 2 * math.pi ) )

	self.gradInput:cmul( gradOutput ) 

	return self.gradInput

end
