# JetPointNet

This repository contains the code used to develop particle-flow models based on pointcloud data for jets data.

## Minimal Installation
For a minimal installation you can first create a conda environment with `conda create --name pointcloud python=3.10`.
Then make use of the `requirements.txt` file and run:

```shell
git clone ssh://git@gitlab.cern.ch:7999/atlas-jetetmiss/pflow/commontools/jetpointnet.git
conda activate pointcloud
pip install -r requirements.txt # --no-cache-dir

```

Please note that `--no-cache-dir` option is suggested.

## Full Working Installation (with pre-commit hooks)
Ensure that you have a python version (3.10 or newer installed on your system). You must also have `tensorflow[and-cuda]` installed.
Then you can clone the repository and run the setup script:

This script will set up the development environment using poetry and install all the necessary dependencies.
Additionally, it will configure pre-commit hooks to ensure consistent style and run test cases.

```shell
git clone ssh://git@gitlab.cern.ch:7999/atlas-jetetmiss/pflow/commontools/jetpointnet.git
make setup-env
```

## Usage

All configuration is done in the `prod/configs/USER_config.toml`.
I config file can also be set in the environment variable `JETPOINTNET_CONFIG_FILE`.
Once a run, either training or data processing is defined, it can be run with:
```shell
make run
```

There are 3 core steps, data processing, chunking, and training.
All are defined in the `prod/configs/USER_config.toml` file.
Chunking is optional and can be skipped if the data is already chunked or sufficiently randomised.

## Next steps:

- Apply garbage collection to the augmented data processing scripts to ensure stills still meet cluster min significance
- Try Mask formers instead of PointNet
- Change batching algorithm for training
- Add CI/CD pipeline for the data processing scripts

## Josh's next steps:
- Add poetryt to ML2 so this runs there
- Test training script
- Migrate training script
- Encode info in file paths for configuration, maybe just use wandb run name??

## Known Issues

- Many... we should probably start a list here
- The early stopping of the train loop can cause the job to crash when enabled
- The augmented data processing scripts work on JZ4 but seem to fail when I run on the server...?

### Authors:
- Dr. Maximilian Swiatlowski (TRIUMF)
- Dr. Luca Clissa (UNIBO)
- Joshua Himmens (TRIUMF/UBC)
- Marko Jovanovic (TRIUMF)
- Jessica Bohm (TRIUMF)
