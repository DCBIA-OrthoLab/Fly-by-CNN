# Create docker image from Dockerfile

1. Clone the repository with tag 2.1

```
	git clone --depth 1 --branch 2.1  https://github.com/DCBIA-OrthoLab/fly-by-cnn.git
```

2. Download the VTK package with EGL support (no X server)

```
	cd fly-by-cnn/src/kubernetes
	wget https://github.com/DCBIA-OrthoLab/fly-by-cnn/releases/download/2.0/vtk8.2.0.zip
```

3. Run docker build

```
	docker build -t flyby:2.0.0 .	
```

# Deploy fly-by-cnn to a kubernetes

## Install the docker engine https://docs.docker.com/get-docker/

## Install the kubectl command https://kubernetes.io/docs/tasks/tools/install-kubectl/

## Install kind https://kind.sigs.k8s.io/docs/user/quick-start/

## Deploy a local cluster using the command

Replace the hostPath by some path in your local machine. 

```
	kind create cluster --config fly-by-cnn/src/kubernetes/kind-config.yaml
```

## Build the fly-by-cnn docker image

```
	docker build -t fly-by-cnn:2.0 fly-by-cnn/src/kubernetes/
```

## Load the docker image into the kind local cluster

```
	kind load docker-image fly-by-cnn:2.0
```

## Volumes, Persistent Volumes, Storage classes 

1. Volumes https://kubernetes.io/docs/concepts/storage/volumes/
2. Persistent Volumes https://kubernetes.io/docs/concepts/storage/persistent-volumes/
3. Storage classes https://kubernetes.io/docs/concepts/storage/storage-classes/

### Create a local-storage class

```
	kubectl apply -f fly-by-cnn/src/kubernetes/local-storage.yaml
```

### Create a volume

```
	kubectl apply -f fly-by-cnn/src/kubernetes/pv.yaml
```

### Create a volume-claim

```
	kubectl apply -f fly-by-cnn/src/kubernetes/pvc.yaml
```

### Run a pod

```
	kubectl apply -f fly-by-cnn/src/kubernetes/fly-by-cnn.yaml
```

### Exec the command in the pod

```
	kubectl exec fly-by-cnn-pod -- python3 /app/fly-by-cnn/src/py/fly_by_features.py
```
