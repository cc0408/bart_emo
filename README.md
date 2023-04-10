#### requirement

pytorch 1.13.1 cuda 11.7

pip install transformers==4.14.1

pip install datasets==2.8.0



#### 训练分类器

mkdir result

python text_classification.py --model facebook/bart-base --dataset sst2 --epochs 1 --finetune True



#### 训练生成模型

python bart.py