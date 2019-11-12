require 'mattorch'

function convertMat( dir ) 

	-- local dir = dir or 'data/mat/yoon'
	local dir = dir or 'data/mat'

	for file in paths.files( dir ) do

		local dirfile = dir .. '/' .. file 

		-- print(dirfile)

		if file ~= '.' and file ~= '..' and paths.dirp( dirfile ) then 

			print( '\ngoing into ' .. dirfile )

			convertMat( dirfile ) 

		elseif dirfile:find('/.+_.*.mat') and not dirfile:find('/._')  then  

			print(dirfile)
			
			local matfile = dirfile
			local t7file  = matfile:gsub('mat','t7')

			print( 'converting ' .. matfile ) 

			data = mattorch.load( matfile )

			os.execute('mkdir -p ' .. paths.dirname( t7file ) ) 



			if t7file:find('corey') then 

				torch.save( t7file, data.responseMatrix )

			else

				torch.save( t7file, data.response_matrix )

			end

			-- if true or  then
			-- 	print('saving to ' .. t7file)
			-- 	print(data )
			-- 	-- print( data.response_matrix )
			-- end

		end

	end 

end
convertMat()
print('done converting')
