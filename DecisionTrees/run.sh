#!/bin/sh
echo "Decision Tree Learning Algorithm - Car Evaluation Task"
python3 DT_categorical.py

echo "Decision Tree Learning Algorithm - Bank Evaluation Task - 'unknown' as attribute value "
python3 DT_numerical.py

echo "Decision Tree Learning Algorithm - Bank Evaluation Task - 'unknown' not as attribute value "
python3 DT_numerical_missing.py