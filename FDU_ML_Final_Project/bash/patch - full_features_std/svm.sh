for c in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
for k in 'linear' 'poly' 'rbf' 'sigmoid'; do
for d in 1 2 3 4 5 6 7 8 9 10; do
python classifier_vG_split_fine_tune.py --c $c --k $k --d $d;
done;
done;
done;