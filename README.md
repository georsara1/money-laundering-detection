This repo is actually a clone of https://github.com/emreakoz/money-laundering-detection. 

All kudos and credits go to this repo and not to me. The reason for creating my repo is that I have changed the Spark framework into Pandas for those who do not have access to a Spark cluster or cannot install Spark locally for whatever reason. 

Note: I cannot guarantee for the methodological soundness of the approach or the validity of the original code. What I can assure is that the Pandas approach delivers the same results as the original Spark implementation (on the specific toy dataset included in the repo). 

Requirements: The code executes correctly on networkx==2.2. I have tested latest versions (e.g. 2.5) and returns "not implemented" error for betweeness centrality. Make sure you have the correct version installed. 

Have fun and enjoy. Feel free to fork to work with Pandas. Please Star the original repo if you find the approach useful, i merely translated the code :)
