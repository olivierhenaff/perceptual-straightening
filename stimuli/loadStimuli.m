% 'NATURAL MOVIES                                                         '
% 'movie_frames: 3 movies x 3 sizes(fovea,parafovea,periphery) x 11 frames'
% 'movie_titles: 04 EGOMOTION, 05 PRAIRIE, 06 CARNEGIE DAM                '
% 'vignettes: 3 vignettes for eeach size                                  '

% loaddir = 'stimuli/yoon/natural_movie_frames.mat';
loaddir = 'stimuli/yoon/synthetic_movie_frames.mat';
t = load( loaddir )

fovea     = []; 
parafovea = [];
periphery = [];

for i = 1:3
	for j = 1:3

		c = cell2mat(t.movie_frames(i,j,:));
		c = reshape(c, [1, size(c)]);

		if     j == 1
			fovea     = [fovea    ; c ];
		elseif j == 2
			parafovea = [parafovea; c ];
		elseif j == 3
			periphery = [periphery; c ];
		end

	end
end

frames_fovea     =     fovea;
frames_parafovea = parafovea;
frames_periphery = periphery;

vignette_fovea     = t.vignettes{1};
vignette_parafovea = t.vignettes{2};
vignette_periphery = t.vignettes{3};

% savedir = 'stimuli/yoon/natural_movie_frames_post';
savedir = 'stimuli/yoon/synthetic_movie_frames_post';
save( savedir )
% save( t, savedir )
% disp(t)

exit