rm -rf Model

python ../scvis/scvis train --data_matrix_file ./Data/crop_sampled_scaled.tsv --out_dir ./Model/ --verbose --verbose_interval 50
