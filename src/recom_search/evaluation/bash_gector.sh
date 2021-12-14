cd $1
# conda init bash
# conda activate run-gector
/home/jcxu/miniconda3/envs/run-gector/bin/python  predict.py --model_path  roberta_1_gectorv2.th \
                   --input_file $2 --output_file $3 --output_cnt_file $4
