
def center_crop(layer, target_size, pad=0):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[
        :, :, (diff_y - pad) : (diff_y + target_size[0] - pad), (diff_x - pad) : (diff_x + target_size[1] - pad)
    ]