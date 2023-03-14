
python train.py \
     --opt sgd \
     --lr 1e-1 \
     --gamma 1e-1 \
     --epoch 60 \
     --decay_epoch 30 \
     --batch_size 128 \
     --val_epoch 5 \
     --weight_decay 5e-4 \
     --nesterov \
     --pre \
     --resnet \
     --train_way 5 \
     --train_support_shot 5 \
     --train_query_shot 15 \
     --train_transform_type 0 \
     --test_way 5 \
     --test_support_shot 1 5 \
     --test_query_shot 16 \
     --alpha 1.0 \
     --weight 0.0 \
     --seed 1 \
     --gpu 0
