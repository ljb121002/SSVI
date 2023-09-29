DATA=cifar100
DENSITY=0.1
for SEED in 0 1 2
do
    CUDA_VISIBLE_DEVICES=$1 python train_cifar.py \
        --seed ${SEED} \
        --data ${DATA} \
        --epochs 200 \
        --lr 0.01 \
        --batch_size 128 \
        --wd 5e-4 \
        --use_bnn \
        --exp_name BNN_${DENSITY}_seed${SEED} \
        --dense_allocation $DENSITY \
done