import convnext

convnext_tiny = convnext.ConvNeXt(
  **convnext.MODEL_CONFIGS.get("tiny"), drop_path_rate=0.0, layer_scale_init_value=0.0,  model_name="convnext_tiny"
)
print(convnext_tiny.summary(expand_nested=True))
