require 'nn' 
require 'nngraph'

function Differences( n ) 

	local inode = nn.Identity()()

	local lin = nn.Linear( n, n^2 )

	lin.bias:zero()
	lin.weight:zero()

	local ind = 0 

	for i = 1, n do 

		for j = 1, n do 

			ind = ind + 1 

			lin.weight[ind][i] = lin.weight[ind][i] + 1
			lin.weight[ind][j] = lin.weight[ind][j] - 1 

		end

	end

	local diff = nn.View( n, n )( lin( inode ) ) 

	local network = nn.gModule({ inode }, { diff })

	return network

end

local diff, parent = torch.class('nn.Differences', 'nn.Sequential')

function diff:__init( n )
	
	parent.__init(self)

	local lin = nn.Linear( n, n^2 )

	lin.bias:zero()
	lin.weight:zero()

	local ind = 0 

	for i = 1, n do 

		for j = 1, n do 

			ind = ind + 1 

			lin.weight[ind][i] = lin.weight[ind][i] + 1
			lin.weight[ind][j] = lin.weight[ind][j] - 1 

		end

	end

	self:add( lin ) 
	self:add( nn.View( n, n ) ) 

end

function testDiff() 

	local n = 11 

	local x = torch.rand( n )

	local n1 = nn.Differences( n ) 
	local n2 =    Differences( n )

	local y1 = n1:updateOutput( x ) 
	local y2 = n2:updateOutput( x ) 

	print('difference between nn and nngraph implementations of Differences:', y1:dist(y2) ) 

end
--testDiff() 

local diff, parent = torch.class('nn.Differences3D', 'nn.Sequential')

function diff:__init( n )
	
	parent.__init(self)

	local lin = nn.Linear( n^2, n^3 )

	lin.bias:zero()
	lin.weight:zero()

	local indo = 0 

	for k = 1, n do 

		for i = 1, n do 

			for j = 1, n do 

				local indi = ( i - 1 ) * n + k 
				local indj = ( j - 1 ) * n + k 
				indo = indo + 1 

				lin.weight[indo][indi] = lin.weight[indo][indi] + 1
				lin.weight[indo][indj] = lin.weight[indo][indj] - 1 

			end

		end

	end

	self:add( lin ) 
	self:add( nn.View( n, n, n ) ) 

end


function testDifferences3Dfwd()

	local ncond = 2 
	local nsmpl = 3

	local net = nn.Sequential()
	net:add( nn.View( ncond, nsmpl*nsmpl ) ) 
	net:add( nn.Differences3D( nsmpl ) )

	x = torch.rand( ncond, nsmpl, nsmpl )

	y1 = net:updateOutput( x )
	y2 = y1:clone():zero()

	for i = 1, nsmpl do 

		for k = 1, nsmpl do 

			for j = 1, nsmpl do 

				local to = y2:select( 2, k ):select( 2, i ):select( 2, j )

				to:copy(    x:select( 2, i ):select( 2, k ) )
				to:add( -1, x:select( 2, j ):select( 2, k ) )

			end

		end

	end

	d = y1:clone():add( -1, y2 ) 

	print( 'difference between output of Differences3D module and algorithmic ground truth' ) 
	print( y1:dist( y2 ) )

end
--testDifferences3Dfwd() 


function viewDifferences()

	n = 5 
	d = nn.Differences( n )
	print( d.weight )

	x = torch.linspace( 0, 1, n )
	y = d:updateOutput( x ) 

	print( x ) 
	print( y ) 

end


local diff, parent = torch.class('nn.CumulativeDifferences', 'nn.Sequential')

function diff:__init( n )
	
	parent.__init(self)

	local lin = nn.Linear( n, (n+1)^2 )
	lin.bias:zero()
	lin.weight:zero()

	local ind = 0 

	for i = 1, n+1 do 

		for j = 1, n+1 do 

			ind = ind + 1 

			local m = math.min( i, j ) 
			local M = math.max( i, j )

			for k = m, M-1 do 

				lin.weight[ind][k] = 1 

			end 

		end

	end

	self:add( lin ) 
	self:add( nn.View( n+1, n+1 ) ) 

end

function viewDifferences()

	n = 5
	d = nn.CumulativeDifferences( n )
	print( d.weight )

	x = torch.linspace( 0, 1, n ):fill( 1 ) 
	y = d:updateOutput( x ) 

	print( x ) 
	print( y ) 

end
--viewDifferences()


local poly, parent = torch.class('nn.Polynomial', 'nn.Linear')

function poly:__init( p, n, c )
	
	parent.__init(self, math.max( 1, p ), n )

	local c = c or math.floor( n / 2 ) 
--	local x = torch.linspace( -c, c, n )
	local x = torch.linspace( 0, 1, n )

	self.bias:zero()
	self.weight:fill( 1 ) 

	for i = 1, p do 

		self.weight:select( 2, i ):copy( x ):pow( p - i + 1 )

	end

end

function testPoly()

	p = nn.Polynomial( 2, 11 ) 

	a = torch.Tensor( 2 ) 
	a[1] = 1 
	a[2] = 0 
	print( p:updateOutput( a ) ) 

	a[1] = 0
	a[2] = 1 
	print( p:updateOutput( a ) ) 

end
--testPoly()




