# Python source

## LinearSubdivisionFilter.py
This code does the sphere subdivion. *For example :* numberOfSubdivisions = 10 -> 1026 ids / numberOfSubdivisions = 5 -> 268 ids 

## predict.py
Does the prediction for the mesh label using the fly-by-cnn idea. It first creates the images with features, does the prediction on this images and projects it in the mesh. You can select the model with the flag --model, the number of subdivisions --numberOfSubdivisions, the output name --out, and your mesh --surf.

## post_process.py
Post preocess is done after the prediction in order to correct the wrong labels. It takes the smallest components in the mesh (< 1000 ids) set there label to -1 and apply a region growing. You select the model --mesh and the output name --out.