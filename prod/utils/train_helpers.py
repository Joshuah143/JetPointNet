import os
from loguru import logger as log


def setup_compute(config: dict):
    gpu_id = config["training"]["infra"]["GPU"]
    log.info(f"Assigning GPU: {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if config["training"]["infra"]["use_cuda_malloc_async"]:
        log.info("Using cuda_malloc_async")
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def verify_model_config(config):
    # TODO: fix this to work with the new config file
    raise NotImplementedError("verify_model_config is not implemented")

    # run_config["training"]["hyperparameters"]["model_params"]["loss_function"] depends on target features

    # note that if you change the output activation function, you must change the loss function
    if (
        baseline_configuration["OUTPUT_ACTIVATION_FUNCTION"] in ["softmax", "sigmoid"]
        and baseline_configuration["OUTPUT_LAYER_SEGMENTATION_CUTOFF"] != 0.5
    ):
        raise Exception("Invalid OUTPUT_LAYER_SEGMENTATION_CUTOFF")
    elif (
        baseline_configuration["OUTPUT_ACTIVATION_FUNCTION"] in ["linear"]
        and baseline_configuration["OUTPUT_LAYER_SEGMENTATION_CUTOFF"] != 0
    ):
        raise Exception("Invalid OUTPUT_LAYER_SEGMENTATION_CUTOFF")
