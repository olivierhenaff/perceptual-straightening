module purge
module load ffmpeg/intel/3.2.2
module load torch/gnu/20170504

# th convert.lua

# for i in {1..15}
# do
	# th analyzeGlobal/bootstrap.lua -seed $i
# done

th analyzeGlobal/bootstrap.lua

