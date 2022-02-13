datasets
	|____ImageNet
		|______imagenet stuff
|______quantization
	  |______3kImages
		|______pruning
  |______3kImages
	|____Pubfig
		|______*.npy
weights
	|____*.h5, *.pt, *.tflite
 
quantization 
	|____ImageNet
		|____attacks.py
|____ModelGen.ipynb
		|____generateImagePerClass.ipynb
		|_____results
			|____WB
			|____PGD
			|____SemiBB
	|____evaluation
		|_____****
|____PubFig
	|_____untargetted
		|______FR_edge.ipynb, FR_server.ipynb, PGD_fr.py, WB_fr.py
 
		|_____results
|____WB
				|____PGD
 
	|_____targetted
		|______******
|______results
|_____Mnist
	|____attacks.ipynb
	|____ModelGen.ipynb
	|____PCA_TSNE.ipynb
	|____results
	
pruning
|____ModelGen.ipynb
|____generateImagePerClass.ipynb
	|____attacks
		|_____DIVA_pqat.py, DIVA_prune.py, PGD_pqat.py, PGD_prune.py
	|____evaluation
	|____results
		|_____prune
		|_____pqat
 
robustness
	|____notebook
		|______ attackandeval.ipynb

