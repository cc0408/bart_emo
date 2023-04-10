#### 训练分类器

mkdir result

python text_classification.py --model facebook/bart-base --dataset sst2 --epochs 1 --finetune True



#### 训练生成模型

python bart.py