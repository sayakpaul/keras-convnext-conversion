import argparse
import os

import tensorflow as tf
import torch
from tensorflow.keras import layers

import convnext_pt
from convnext import ConvNeXt
from model_configs import get_model_config

torch.set_grad_enabled(False)

DATASET_TO_CLASSES = {
    "imagenet-1k": 1000,
    "imagenet-21k": 21841,
}
MODEL_TO_METHOD = {
    "convnext_tiny": convnext_pt.convnext_tiny,
    "convnext_small": convnext_pt.convnext_small,
    "convnext_base": convnext_pt.convnext_base,
    "convnext_large": convnext_pt.convnext_large,
    "convnext_xlarge": convnext_pt.convnext_xlarge,
}
TF_MODEL_ROOT = "keras-applications/convnext"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the PyTorch pre-trained ConvNeXt weights to TensorFlow."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="imagenet-1k",
        type=str,
        help="Name of the pretraining dataset.",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="convnext_tiny",
        type=str,
        choices=[
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
            "convnext_xlarge",
        ],
        help="Name of the ConvNeXt model variant.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default=224,
        type=int,
        help="Image resolution.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        default="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        type=str,
        help="URL of the checkpoint to be loaded.",
    )
    parser.add_argument(
        "-t",
        "--include-top",
        action="store_true",
        help="Whether to include the classification top.",
    )
    return vars(parser.parse_args())


def main(args):
    print(f'Model: {args["model_name"]}')
    print(f'Image resolution: {args["resolution"]}')
    print(f'Dataset: {args["dataset"]}')
    print(f'Checkpoint URL: {args["checkpoint_path"]}')

    print("Instantiating PyTorch model and populating weights...")
    model_method = MODEL_TO_METHOD[args["model_name"]]
    convnext_model_pt = model_method(
        args["checkpoint_path"], num_classes=DATASET_TO_CLASSES[args["dataset"]]
    )
    convnext_model_pt.eval()

    print("Instantiating TensorFlow model...")
    model_config = get_model_config(args["model_name"])
    model_name = args["model_name"]

    convnext_model_tf = ConvNeXt(
        model_name=model_name,
        default_size=args["resolution"],
        classes=DATASET_TO_CLASSES[args["dataset"]],
        depths=model_config.depths,
        projection_dims=model_config.dims,
        include_top=args["include_top"],
    )

    if args["include_top"]:
        assert convnext_model_tf.count_params() == sum(
            p.numel() for p in convnext_model_pt.parameters()
        )
    print("TensorFlow model instantiated, populating pretrained weights...")

    # Fetch the pretrained parameters.
    param_list = list(convnext_model_pt.parameters())
    model_states = convnext_model_pt.state_dict()
    state_list = list(model_states.keys())

    # Stem block.
    stem_block = convnext_model_tf.get_layer(f"{model_name}_stem")

    for layer in stem_block.layers:
        if isinstance(layer, layers.Conv2D):
            layer.kernel.assign(
                tf.Variable(param_list[0].numpy().transpose(2, 3, 1, 0))
            )
            layer.bias.assign(tf.Variable(param_list[1].numpy()))
        elif isinstance(layer, layers.LayerNormalization):
            layer.gamma.assign(tf.Variable(param_list[2].numpy()))
            layer.beta.assign(tf.Variable(param_list[3].numpy()))

    # Downsampling layers.
    for i in range(3):
        downsampling_block = convnext_model_tf.get_layer(model_name + "_downsampling_block_" + str(i))
        pytorch_layer_prefix = f"downsample_layers.{i + 1}"

        for l in downsampling_block.layers:
            if isinstance(l, layers.LayerNormalization):
                l.gamma.assign(
                    tf.Variable(
                        model_states[f"{pytorch_layer_prefix}.0.weight"].numpy()
                    )
                )
                l.beta.assign(
                    tf.Variable(model_states[f"{pytorch_layer_prefix}.0.bias"].numpy())
                )
            elif isinstance(l, layers.Conv2D):
                l.kernel.assign(
                    tf.Variable(
                        model_states[f"{pytorch_layer_prefix}.1.weight"]
                        .numpy()
                        .transpose(2, 3, 1, 0)
                    )
                )
                l.bias.assign(
                    tf.Variable(model_states[f"{pytorch_layer_prefix}.1.bias"].numpy())
                )

    # ConvNeXt stages.
    num_stages = 4

    for m in range(num_stages):
        stage_name = model_name + f"_stage_{m}"
        num_blocks = len(convnext_model_tf.get_layer(stage_name).layers)

        for i in range(num_blocks):
            stage_block = convnext_model_tf.get_layer(stage_name).get_layer(
                f"{stage_name}_block_{i}"
            )
            stage_prefix = f"stages.{m}.{i}"

            for j, layer in enumerate(stage_block.layers):
                if isinstance(layer, layers.Conv2D):
                    layer.kernel.assign(
                        tf.Variable(
                            model_states[f"{stage_prefix}.dwconv.weight"]
                            .numpy()
                            .transpose(2, 3, 1, 0)
                        )
                    )
                    layer.bias.assign(
                        tf.Variable(model_states[f"{stage_prefix}.dwconv.bias"].numpy())
                    )
                elif isinstance(layer, layers.Dense):
                    if j == 2:
                        layer.kernel.assign(
                            tf.Variable(
                                model_states[f"{stage_prefix}.pwconv1.weight"]
                                .numpy()
                                .transpose()
                            )
                        )
                        layer.bias.assign(
                            tf.Variable(
                                model_states[f"{stage_prefix}.pwconv1.bias"].numpy()
                            )
                        )
                    elif j == 4:
                        layer.kernel.assign(
                            tf.Variable(
                                model_states[f"{stage_prefix}.pwconv2.weight"]
                                .numpy()
                                .transpose()
                            )
                        )
                        layer.bias.assign(
                            tf.Variable(
                                model_states[f"{stage_prefix}.pwconv2.bias"].numpy()
                            )
                        )
                elif isinstance(layer, layers.LayerNormalization):
                    layer.gamma.assign(
                        tf.Variable(model_states[f"{stage_prefix}.norm.weight"].numpy())
                    )
                    layer.beta.assign(
                        tf.Variable(model_states[f"{stage_prefix}.norm.bias"].numpy())
                    )

            stage_block.gamma.assign(
                tf.Variable(model_states[f"{stage_prefix}.gamma"].numpy())
            )

    # Final LayerNormalization layer and classifier head.
    if args["include_top"]:
        convnext_model_tf.layers[-2].gamma.assign(
            tf.Variable(model_states[state_list[-4]].numpy())
        )
        convnext_model_tf.layers[-2].beta.assign(
            tf.Variable(model_states[state_list[-3]].numpy())
        )

        convnext_model_tf.layers[-1].kernel.assign(
            tf.Variable(model_states[state_list[-2]].numpy().transpose())
        )
        convnext_model_tf.layers[-1].bias.assign(
            tf.Variable(model_states[state_list[-1]].numpy())
        )
    print("Weight population successful, serializing TensorFlow model...")
    model_name = f"{model_name}_in21k" if args["dataset"] == "imagenet-21k" else model_name
    model_name = f"{model_name}.h5" if args["include_top"] else f"{model_name}_notop.h5"
    save_path = os.path.join(TF_MODEL_ROOT, model_name)
    convnext_model_tf.save_weights(save_path)
    print(f"TensorFlow model serialized to: {save_path}...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
