import argparse





def get_node_params():

    parser = argparse.ArgumentParser(description='Run GreeCo on Node Classification')

    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--use_params', action='store_true', help='Use best parameters')
    parser.add_argument('--setting', type=str, default='trans', choices=['trans', 'ind'], help='Transductive or Inductive')
    parser.add_argument('--device', type=int, default=0)

    # Only valid in transductive setting

    parser.add_argument('--num_splits', type=int, default=10, help='Number of splits in transductive testing')
    parser.add_argument('--cluster', action='store_true', help='Cluster nodes')
    parser.add_argument('--get_ncut', action='store_true', help='Compute normalized cut')
    parser.add_argument('--get_smooth', action='store_true', help='Compute smoothness value')

    # Only valid in inductive setting
    parser.add_argument('--ind_rate', type=float, default=0.2, help='Inductive split ratio')

    # Only valid in supervised GreeCo
    parser.add_argument('--clf_lambda', type=float, default=1.0, help='Classification loss weight')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--verbose', type=int, default=100, help='Verbose')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size')

    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--pred_dim', type=int, default=512, help='Prediction dimension')

    parser.add_argument('--enc_layers', type=int, default=2, help='Number of layers in MLP encoder')
    parser.add_argument('--enc_norm', type=str, default='batch', help='Normalization in MLP encoder')
    parser.add_argument('--enc_drop', type=float, default=0.0, help='Dropout rate in MLP encoder')
    parser.add_argument('--res_enc', action='store_true', default=True, help='Residual connection in MLP encoder')

    parser.add_argument('--proj_layers', type=int, default=2, help='Number of layers in non-parametric aggregator')
    parser.add_argument('--proj_norm', type=str, default='batch', help='Normalization in non-parametric aggregator')
    parser.add_argument('--proj_drop', type=float, default=0.0, help='Dropout rate in non-parametric aggregator')
    parser.add_argument('--aggr_norm', type=str, default='gcn', help='Aggregation type in non-parametric aggregator')

    parser.add_argument('--pred_layers', type=int, default=2, help='Number of layers in predictor')
    parser.add_argument('--pred_norm', type=str, default='batch', help='Normalization in predictor')
    parser.add_argument('--pred_drop', type=float, default=0.0, help='Dropout rate in predictor')

    parser.add_argument('--lr_learning_rate', type=float, default=0.01, help='Learning rate for downstream task head')
    parser.add_argument('--lr_weight_decay', type=float, default=0.0, help='Weight decay for downstream task head')
    parser.add_argument('--lr_batch_size', type=int, default=0, help='Batch size for downstream task head')
    parser.add_argument('--lr_epochs', type=int, default=5000, help='Number of epochs for downstream task head')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')

    parser.add_argument('--feature_mask', type=float, default=0.25, help='Feature mask rate')
    parser.add_argument('--edge_mask', type=float, default=0.25, help='Edge mask rate')

    parser.add_argument('--aug_rounds', type=int, default=1, help='Number of augmentation rounds')
    parser.add_argument('--recon_lambda', type=float, default=10, help='Reconstruction loss weight')

    parser.add_argument('--eta', type=float, default=1.0, help='Scaling factor for bootstrap loss')
    parser.add_argument('--aux_pos_ratio', type=int, default=0, help='Extra positive ratio for bootstrap loss')
    parser.add_argument('--use_scheduler', action='store_true', default=True, help='Use scheduler')

    # Test model robustness
    parser.add_argument('--train_ratio', type=float, default=0.1, help='Train ratio')
    parser.add_argument('--feature_noise', type=float, default=0.0, help='Feature noise rate')
    parser.add_argument('--edge_noise', type=float, default=0.0, help='Edge noise rate')

    args = parser.parse_args()

    return vars(args)





def get_graph_params():

    parser = argparse.ArgumentParser(description='Run GreeCo on Graph Classification')

    parser.add_argument('--dataset', type=str, default='mutag', help='Dataset name')

    parser.add_argument('--use_params', action='store_true', help='Use best parameters')

    parser.add_argument('--device', type=int, default=0)



    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

    parser.add_argument('--verbose', type=int, default=10, help='Verbose')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument('--deg4feat', action='store_true', default=True, help='Use degree to produce one-hot node features')

    parser.add_argument('--pooling', type=str, default='sum', choices=['sum', 'mean', 'max'], help='Pooling method')



    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')

    parser.add_argument('--pred_dim', type=int, default=512, help='Prediction dimension')



    parser.add_argument('--enc_layers', type=int, default=2, help='Number of layers in MLP encoder')

    parser.add_argument('--enc_norm', type=str, default='batch', help='Normalization in MLP encoder')

    parser.add_argument('--enc_drop', type=float, default=0.0, help='Dropout rate in MLP encoder')

    parser.add_argument('--res_enc', action='store_true', default=True, help='Residual connection in MLP encoder')



    parser.add_argument('--proj_layers', type=int, default=2, help='Number of layers in non-parametric aggregator')

    parser.add_argument('--proj_norm', type=str, default='batch', help='Normalization in non-parametric aggregator')

    parser.add_argument('--proj_drop', type=float, default=0.0, help='Dropout rate in non-parametric aggregator')

    parser.add_argument('--aggr_norm', type=str, default='gcn', help='Aggregation type in non-parametric aggregator')



    parser.add_argument('--pred_layers', type=int, default=2, help='Number of layers in predictor')

    parser.add_argument('--pred_norm', type=str, default='batch', help='Normalization in predictor')

    parser.add_argument('--pred_drop', type=float, default=0.0, help='Dropout rate in predictor')



    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')



    parser.add_argument('--feature_mask', type=float, default=0.25, help='Feature mask rate')

    parser.add_argument('--edge_mask', type=float, default=0.25, help='Edge mask rate')



    parser.add_argument('--aug_rounds', type=int, default=1, help='Number of augmentation rounds')

    parser.add_argument('--recon_lambda', type=float, default=10, help='Reconstruction loss weight')



    parser.add_argument('--eta', type=float, default=1.0, help='Scaling factor for bootstrap loss')

    parser.add_argument('--aux_pos_ratio', type=int, default=0, help='Extra positive ratio for bootstrap loss')

    parser.add_argument('--use_scheduler', action='store_true', default=True, help='Use scheduler')



    args = parser.parse_args()

    return vars(args)

