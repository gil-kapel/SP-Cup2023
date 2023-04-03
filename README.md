# spcup
1. Add dataset to data directory, it will not be uploaded to giuhub
2. If needed, Build docker image
```bash
docker build --tag spcup:latest ./docker/
```
3. Run container
```
docker run --name=torch --network=host --ipc=host --gpus=all -it --rm -v "$PWD":"/spcup" spcup
```