
生成数据集
`python generate_dataset.py --input_folder data/datasets/BSRN_train/test --size 256`

训练
`python main_vdsr.py --batchSize 64 --cuda --tag plane`



`pip install opencv-python`
## 6 个

`python generate_dataset.py --input_folder data/datasets/BSRN_train/car_train --size 256`
`python main_vdsr.py --batchSize 64 --cuda --tag car`

`python generate_dataset.py --input_folder data/datasets/BSRN_train/cat_train --size 256`
`python main_vdsr.py --batchSize 64 --cuda --tag cat`

`python generate_dataset.py --input_folder data/datasets/BSRN_train/dog_train --size 256`
`python main_vdsr.py --batchSize 64 --cuda --tag dog`

`python generate_dataset.py --input_folder data/datasets/BSRN_train/mixure_train --size 256`
`python main_vdsr.py --batchSize 64 --cuda --tag mixure`

`python generate_dataset.py --input_folder data/datasets/BSRN_train/people_train --size 256`
`python main_vdsr.py --batchSize 64 --cuda --tag people`

`python generate_dataset.py --input_folder data/datasets/BSRN_train/plane_train --size 256`
`python main_vdsr.py --batchSize 64 --cuda --tag plane`


## test
img
- crop -> label
- get mask
- rrx4
- irx4 -> input
- seg with mask -> inputs
- sr
- blend


- process_img: python main.py --mode process_img --opt tools/process_img2/options/option_set_02_vdsrwithseg/options.yml
- sr: python my_test.py
- blend: python main.py --mode blend
- blend: python main.py --mode psnr

python main.py --mode all --opt tools/process_img2/options/option_set_02_vdsrwithseg/options.yml  
python main.py --mode all --opt tools/process_img2/options/option_set_03_vdsrwithsegJiekou/options.yml 