jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --to notebook --inplace Traffic_Sign_Classifier_Submission.ipynb
echo "Done 1024" | mail -s "Done 1024" -u atchirc@marvell.com
