ref=$1
gen=$2

echo "Evaluating FID and KID for ${gen}"
python eval_fid_kid.py $ref $gen
echo "Evaluating GIQA-QS ${gen}"
python eval_giqa.py $ref $gen
echo "Evaluating GIQA-DS ${gen}"
python eval_giqa.py $gen $ref
echo "Evaluating CLIP score ${gen}"
python eval_clip_score.py $gen
