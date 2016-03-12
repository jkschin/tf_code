# Description:
# Example TensorFlow models for CIFAR-10

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "cnn_input",
    srcs = ["cnn_input.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "cnn_input_test",
    srcs = ["cnn_input_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cnn_input",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "cnn",
    srcs = ["cnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cnn_input",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "cnn_eval",
    srcs = [
        "cnn_eval.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":cnn",
    ],
)

py_binary(
    name = "cnn_train",
    srcs = [
        "cnn_train.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":cnn",
    ],
)

py_binary(
    name = "cnn_multi_gpu_train",
    srcs = [
        "cnn_multi_gpu_train.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":cnn",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
