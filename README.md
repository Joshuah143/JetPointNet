
## Installation
The current version has been tested with the following environment settings:

```
python 3.10.14
awkward 2.6.4
uproot 5.3.7
numpy 1.26.4
pandas 2.2.2
tensorflow 2.16.1
keras 3.3.3
```

For a minimal installation you can first create a conda environment with `conda create --name pointcloud python=3.10`. 
Then make use of the `requirements.txt` file and run:

```
conda activate pointcloud
pip install -r requirements.txt # --no-cache-dir

```

Please note that `--no-cache-dir` option is suggested.