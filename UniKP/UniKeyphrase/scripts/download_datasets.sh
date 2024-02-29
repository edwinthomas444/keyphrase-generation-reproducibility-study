# move to data directory and download datasets here
cd data

# duc
cd duc
wget -O duc_test.json https://huggingface.co/datasets/memray/duc/resolve/main/test.json
cd ..

# inspec
cd inspec
wget -O inspec_test.json https://huggingface.co/datasets/memray/inspec/resolve/main/test.json
wget -O inspec_valid.json https://huggingface.co/datasets/memray/inspec/resolve/main/valid.json
cd ..

# kp20k
cd kp20k
wget -O kp20k_test.json https://huggingface.co/datasets/memray/kp20k/resolve/main/test.json
wget -O kp20k_valid.json https://huggingface.co/datasets/memray/kp20k/resolve/main/valid.json
wget -O kp20k_train.json https://huggingface.co/datasets/memray/kp20k/resolve/main/train.json
cd ..

# krapvin
cd krapivin
wget -O krapvin_test.json https://huggingface.co/datasets/memray/krapivin/resolve/main/test.json
wget -O krapvin_valid.json https://huggingface.co/datasets/memray/krapivin/resolve/main/valid.json
cd ..

# nus
cd nus
wget -O nus_test.json https://huggingface.co/datasets/memray/nus/resolve/main/test.json
cd ..

# semeval
cd semeval
wget -O semeval_test.json https://huggingface.co/datasets/memray/semeval/resolve/main/test.json
wget -O semeval_valid.json https://huggingface.co/datasets/memray/semeval/resolve/main/valid.json
cd ..

# stackexchange
cd stackexchange
wget -O stackexchange_test.json https://huggingface.co/datasets/memray/stackexchange/resolve/main/test.json
wget -O stackexchange_valid.json https://huggingface.co/datasets/memray/stackexchange/resolve/main/valid.json
wget -O stackexchange_train.json https://huggingface.co/datasets/memray/stackexchange/resolve/main/train.json
# cd ..

# openkp
cd openkp
wget -O openkp_train.json https://huggingface.co/datasets/memray/openkp/resolve/main/train.json
wget -O openkp_valid.json https://huggingface.co/datasets/memray/openkp/resolve/main/valid.json
wget -O openkp_test.json https://huggingface.co/datasets/memray/openkp/resolve/main/test.json
cd ..

# kptimes
cd kptimes
wget -O kptimes_train.json https://huggingface.co/datasets/memray/kptimes/resolve/main/train.json
wget -O kptimes_valid.json https://huggingface.co/datasets/memray/kptimes/resolve/main/valid.json
wget -O kptimes_test.json https://huggingface.co/datasets/memray/kptimes/resolve/main/test.json
cd ..