from collections import namedtuple


StyleTransferModelConfig = namedtuple(
    'StyleTransferModelConfig', 'dim_y dim_z dim_emb n_layers max_seq_length filter_sizes n_filters')

style_transfer = StyleTransferModelConfig(
    dim_y=200, dim_z=500, dim_emb=100, n_layers=1, max_seq_length=20, filter_sizes=[1, 2, 3, 4, 5], n_filters=128)
