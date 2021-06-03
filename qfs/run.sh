python qfs_ctrl_baseline.py --path='/mnt/data0/ojas/QFSumm/data/tokenized/eq_100_rescue' --name='eq_100_rescue' --device='cuda:0'

python qfs_ctrl_baseline.py --path='/mnt/data0/ojas/QFSumm/data/tokenized/eq_100_geo' --name='eq_100_geo' --device='cuda:0'

python qfs_ctrl_baseline.py --path='/mnt/data0/ojas/QFSumm/data/tokenized/fraud_100_penalty' --name='fraud_100_penalty' --device='cuda:1'

python qfs_ctrl_baseline.py --path='/mnt/data0/ojas/QFSumm/data/tokenized/fraud_100_nature' --name='fraud_100_nature' --device='cuda:1'

python qfs_ctrl_baseline.py --path='/mnt/data0/ojas/QFSumm/data/tokenized/eq_100_rescue' --name='eq_100_rescue_unc' --device='cuda:2' --keyword_suffix='none'

python qfs_ctrl_baseline.py --path='/mnt/data0/ojas/QFSumm/data/tokenized/eq_100_geo' --name='eq_100_geo_unc' --device='cuda:2' --keyword_suffix='none'

python qfs_ctrl_baseline.py --path='/mnt/data0/ojas/QFSumm/data/tokenized/fraud_100_penalty' --name='fraud_100_penalty_unc' --device='cuda:3' --keyword_suffix='none'

python qfs_ctrl_baseline.py --path='/mnt/data0/ojas/QFSumm/data/tokenized/fraud_100_nature' --name='fraud_100_nature_unc' --device='cuda:3' --keyword_suffix='none'

