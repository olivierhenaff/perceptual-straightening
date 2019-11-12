#!/bin/bash

for r in 1
do

	echo
	echo repeat number      n$r

	for i in 1 #2 3 4 5 6
	do 

		sh clean.sh 

		if [[ $i > 1 || $r > 1 ]];
		then
			while [[ $(squeue --noheader -j $jobID 2> /dev/null | wc -l) > 0 ]]
			do
				echo sleeping
				sleep 60
			done
		fi

		echo launching perceptual n$i
		jobID=$(sbatch analyzeGlobal/batch/bootstrapPerceptual.s $r | awk '{print $NF}')
		jobID=$(sbatch --dependency=afterany:$jobID analyzeGlobal/batch/collectPerceptual.s $r | awk '{print $NF}')

		while [[ $(squeue --noheader -j $jobID 2> /dev/null | wc -l) > 0 ]]
		do
			echo sleeping
			sleep 60
		done

		sh clean.sh 

		echo launching pixel      n$i
		jobID=$(sbatch analyzeGlobal/batch/bootstrapPixel.s $r | awk '{print $NF}')
		jobID=$(sbatch --dependency=afterany:$jobID analyzeGlobal/batch/collectPixel.s $r | awk '{print $NF}')

	done

done