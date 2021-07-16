#!/bin/sh

Help()
{
	echo
	echo "Create output folders for each cluster of a fiber bundle from the input atlas"
	echo
	echo "options:"
	echo "-h, --help Print this Help."
	echo "--input Input directory"
	echo "--output Output directory"
	echo "--script fly_by_features script to extract fibers"
}

CreateArchitecture(){

	paths_tract=$input_path_dir"/T_*"
	for path_tract in $paths_tract
	do
		tract_name=`basename $path_tract`
		path_dir_tract=$output_path_dir"/"$tract_name
		mkdir $path_dir_tract

		paths_clusters=$path_tract"/Data/*.vtk"
		for path_cluster in $paths_clusters
		do
			clusterName=`basename $path_cluster | cut -f1 -d'.' `
			path_dir_cluster=$path_dir_tract"/"$clusterName
			mkdir $path_dir_cluster


			commandLine="python3 $flybyfeature_path --surf $path_cluster --fiberBundle 1 --spiral 64 --out $path_dir_cluster --nbFiber 20 --scale_factor 0.97 --translate -90.65 -126.0 -72.0 --shape 202 242 202"
			eval $commandLine



		done
		# echo
	done
}

###
# Main 
###

input_path_dir=""
output_path_dir=""
previous=""
flybyfeature_path=""

for var in "$@"
do
	if [ "$var" == "-h" ] || [ "$var" == "--help" ]; then
		Help
		exit;
	fi

	if [ "$previous" == "--input" ]; then
		input_path_dir=$var
	fi

	if [ "$previous" == "--output" ]; then
		output_path_dir=$var
	fi

	if [ "$previous" == "--script" ]; then
		flybyfeature_path=$var
	fi
	previous=$var
done


if [ output_path_dir != "" ] && [ input_path_dir != "" ];	then
	CreateArchitecture
fi
