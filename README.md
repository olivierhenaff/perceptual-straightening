# perceptual-straightening

Repository for inferring perceptual trajectories and their curvature from discriminability data. Requires Torch7 (see http://torch.ch/docs/getting-started.html#) and randomkit (luarocks install randomkit). 

Once installed, run analyzeGlobal/bootstrap.lua (i.e. "th analyzeGlobal/bootstrap.lua" from the main directory) to launch analysis. As described therein, the bootstrap.lua script supports 3 different modes which are controlled by the --seed flag you can pass (or change the default of). The first phase (seed>0) consists of the analysis proper, which then saves the results in an intermediate file. These files (typically, results from different bootstrapped samples) are then collected in the second phase (seed=-1), and summary statistics are saved in a second file. These summaries are then read in the final phase (seed=0), for plotting etc.
