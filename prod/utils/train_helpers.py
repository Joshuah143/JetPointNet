import os


def setup_compute(config: dict):
    gpu_id = config["training"]["infra"]["gpu_id"]
    print(f"Assigning GPU: {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if config["training"]["infra"]["use_cuda_malloc_async"]:
        print("Using cuda_malloc_async")
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def verify_model_config(config):
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
